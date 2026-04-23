# =========================
# XAI Step 1: SHAP using GradientExplainer
# =========================
!pip -q install shap

import os, json, copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import shap

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score

import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR_RND = os.path.join(PROJECT_ROOT, "prepared_upfall")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "outputs", "xai")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Load data + predictions from Step 3
# =========================
X          = np.load(os.path.join(DATA_DIR, "X_windows.npy"))
y          = np.load(os.path.join(DATA_DIR, "y.npy"))
groups     = np.load(os.path.join(DATA_DIR, "groups.npy"))
activities = np.load(os.path.join(DATA_DIR, "activities.npy"))
with open(os.path.join(DATA_DIR, "metadata.json")) as f:
    feature_names = json.load(f)["feature_names"]
with open(os.path.join(DATA_DIR, "loso_folds.json")) as f:
    folds = json.load(f)

preds_df = pd.read_csv(os.path.join(PROJECT_ROOT, "outputs", "diagnostics",
                                    "loso_per_window_predictions.csv"))

# =========================
# Model (same as before)
# =========================
HIDDEN_SIZE, NUM_LAYERS, BIDIRECTIONAL, DROPOUT = 128, 2, True, 0.3

class TemporalAttention(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.attn = nn.Linear(d, 1)
    def forward(self, x):
        w = F.softmax(self.attn(x).squeeze(-1), dim=1).unsqueeze(-1)
        return (x * w).sum(dim=1), w.squeeze(-1)   # return weights too!

class FallLSTM(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        d = HIDDEN_SIZE * (2 if BIDIRECTIONAL else 1)
        self.lstm = nn.LSTM(input_size, HIDDEN_SIZE, NUM_LAYERS, batch_first=True,
                            dropout=DROPOUT, bidirectional=BIDIRECTIONAL)
        self.ln   = nn.LayerNorm(d)
        self.att  = TemporalAttention(d)
        self.drop = nn.Dropout(DROPOUT)
        self.head = nn.Sequential(nn.Linear(d, 64), nn.ReLU(),
                                  nn.Dropout(0.4), nn.Linear(64, 1))
        self.last_attn = None
    def forward(self, x):
        h, _ = self.lstm(x)
        h    = self.ln(h)
        pooled, attn_weights = self.att(h)
        self.last_attn = attn_weights
        return self.head(self.drop(pooled))

# =========================
# Pick a fold to explain — we'll use S13 (failure case) and S03 (best case)
# =========================
def train_one_fold(fold, seed=42):
    """Retrain one fold quickly just to get a trained model for XAI."""
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    train_idx = np.array(fold["train_indices"])
    test_idx  = np.array(fold["test_indices"])

    # Per-channel normalization (same as baseline)
    X_tr = np.empty_like(X[train_idx]); X_te = np.empty_like(X[test_idx])
    for c in range(X.shape[2]):
        sc = StandardScaler()
        X_tr[:,:,c] = sc.fit_transform(X[train_idx][:,:,c])
        X_te[:,:,c] = sc.transform(X[test_idx][:,:,c])

    y_tr = y[train_idx]; y_te = y[test_idx]

    # Val split
    sss = StratifiedShuffleSplit(1, test_size=0.15, random_state=seed)
    tr, val = next(sss.split(X_tr, y_tr))
    Xa, ya, Xv, yv = X_tr[tr], y_tr[tr], X_tr[val], y_tr[val]

    # Class weight
    n_pos = max(int((ya == 1).sum()), 1)
    pw = torch.tensor([min(int((ya == 0).sum()) / n_pos, 3.0)],
                      dtype=torch.float32).to(DEVICE)

    # Training
    class DS(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        def __len__(self): return len(self.X)
        def __getitem__(self, i): return self.X[i], self.y[i]

    tl = DataLoader(DS(Xa, ya), batch_size=64, shuffle=True)
    vl = DataLoader(DS(Xv, yv), batch_size=64)

    model = FallLSTM(X.shape[2]).to(DEVICE)
    crit  = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_f1, best_state, pc = -1, None, 0
    for ep in range(100):
        model.train()
        for xb, yb in tl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            xb = xb + torch.randn_like(xb) * 0.02
            opt.zero_grad(); crit(model(xb), yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval(); vp = []
        with torch.no_grad():
            for xb,_ in vl: vp.extend(torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy().flatten())
        f1 = f1_score(yv, (np.array(vp) > 0.5).astype(int), zero_division=0)
        if f1 > best_f1: best_f1, best_state, pc = f1, copy.deepcopy(model.state_dict()), 0
        else:
            pc += 1
            if pc >= 12: break

    model.load_state_dict(best_state)
    return model, X_tr, X_te, train_idx, test_idx

# Pick fold for S13 (worst failing case)
fold_s13 = next(f for f in folds if f["test_subject"] == 13)
print("Training S13 fold for XAI...")
model, X_tr_norm, X_te_norm, train_idx, test_idx = train_one_fold(fold_s13)
print("Done.")

# =========================
# SHAP using GradientExplainer (proper tool for PyTorch)
# =========================
# Background: a small sample of training data
rng = np.random.RandomState(42)
bg_idx = rng.choice(len(X_tr_norm), 100, replace=False)
background = torch.tensor(X_tr_norm[bg_idx], dtype=torch.float32).to(DEVICE)

# FIX: disable cuDNN RNN optimization so backward works in eval mode
torch.backends.cudnn.enabled = False

model.eval()
explainer = shap.GradientExplainer(model, background)

# Pick windows to explain: one correct fall, one missed fall, one FP lying, one correct lying
s13_preds = preds_df[preds_df["subject"] == 13]

def pick_one(df):
    if len(df) == 0: return None
    return df.iloc[0]["window"]

candidates = {
    "correct_fall":    pick_one(s13_preds[(s13_preds.y_true==1) & (s13_preds.y_pred==1)]),
    "missed_fall":     pick_one(s13_preds[(s13_preds.y_true==1) & (s13_preds.y_pred==0)]),
    "false_positive":  pick_one(s13_preds[(s13_preds.y_true==0) & (s13_preds.y_pred==1)]),
    "correct_reject":  pick_one(s13_preds[(s13_preds.y_true==0) & (s13_preds.y_pred==0)]),
}

# Map global window indices to local indices in X_te_norm
global_to_local = {int(g): i for i, g in enumerate(test_idx)}

# =========================
# Compute SHAP and visualize
# =========================
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

for i, (tag, global_idx) in enumerate(candidates.items()):
    if global_idx is None:
        axes[i].set_title(f"{tag}: no example")
        continue
    local_idx = global_to_local[int(global_idx)]
    x_exp = torch.tensor(X_te_norm[local_idx:local_idx+1], dtype=torch.float32).to(DEVICE)

    shap_values = explainer.shap_values(x_exp, nsamples=200)
    sv = np.array(shap_values).squeeze()   # (T, C)

    # True activity and prediction info
    row = s13_preds[s13_preds.window == int(global_idx)].iloc[0]
    activity = int(activities[int(global_idx)])

    ACTIVITY_NAMES = {
        1:"fall/hands", 2:"fall/knees", 3:"fall/back", 4:"fall/sideways",
        5:"fall/sitting", 6:"walking", 7:"standing", 8:"sitting",
        9:"picking up", 10:"jumping", 11:"lying"
    }

    im = axes[i].imshow(sv.T, aspect="auto", cmap="RdBu_r",
                        vmin=-np.abs(sv).max(), vmax=np.abs(sv).max())
    axes[i].set_title(f"{tag}\nw{int(global_idx)} activity={ACTIVITY_NAMES.get(activity, activity)} "
                      f"true={int(row.y_true)} pred={int(row.y_pred)} prob={row.prob:.2f}",
                      fontsize=10)
    axes[i].set_xlabel("Time step")
    axes[i].set_ylabel("Channel")
    axes[i].set_yticks(range(0, 30, 3))
    axes[i].set_yticklabels([feature_names[c] for c in range(0, 30, 3)], fontsize=7)
    plt.colorbar(im, ax=axes[i], label="SHAP")

plt.suptitle("SHAP explanations — Subject 13 (worst LOSO failure)", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "shap_s13_four_cases.png"), dpi=120)
plt.show()
print(f"Saved → {os.path.join(OUTPUT_DIR, 'shap_s13_four_cases.png')}")