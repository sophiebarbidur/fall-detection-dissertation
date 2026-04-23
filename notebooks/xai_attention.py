# =========================
# Attention visualization — LOSO model (S13 fold)
# =========================

import os, json, copy, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR_RND = os.path.join(PROJECT_ROOT, "prepared_upfall")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "outputs", "xai")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = False  # needed for gradient-based XAI compatibility

# =========================
# Load
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

ACTIVITY_NAMES = {
    1:"fall/hands", 2:"fall/knees", 3:"fall/back", 4:"fall/sideways",
    5:"fall/sitting", 6:"walking", 7:"standing", 8:"sitting",
    9:"picking up", 10:"jumping", 11:"lying"
}

# =========================
# Model (with last_attn exposed)
# =========================
HIDDEN_SIZE, NUM_LAYERS, BIDIRECTIONAL, DROPOUT = 128, 2, True, 0.3

class TemporalAttention(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.attn = nn.Linear(d, 1)
    def forward(self, x):
        w = F.softmax(self.attn(x).squeeze(-1), dim=1).unsqueeze(-1)
        return (x * w).sum(dim=1), w.squeeze(-1)

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
# Train S13 LOSO fold
# =========================
def train_s13_fold(seed=42):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    fold = next(f for f in folds if f["test_subject"] == 13)
    train_idx = np.array(fold["train_indices"])
    test_idx  = np.array(fold["test_indices"])

    X_tr = np.empty_like(X[train_idx]); X_te = np.empty_like(X[test_idx])
    for c in range(X.shape[2]):
        sc = StandardScaler()
        X_tr[:,:,c] = sc.fit_transform(X[train_idx][:,:,c])
        X_te[:,:,c] = sc.transform(X[test_idx][:,:,c])
    y_tr = y[train_idx]; y_te = y[test_idx]

    sss = StratifiedShuffleSplit(1, test_size=0.15, random_state=seed)
    tr, val = next(sss.split(X_tr, y_tr))
    Xa, ya, Xv, yv = X_tr[tr], y_tr[tr], X_tr[val], y_tr[val]

    n_pos = max(int((ya == 1).sum()), 1)
    pw = torch.tensor([min(int((ya == 0).sum()) / n_pos, 3.0)],
                      dtype=torch.float32).to(DEVICE)

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
            for xb, _ in vl: vp.extend(torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy().flatten())
        f1 = f1_score(yv, (np.array(vp) > 0.5).astype(int), zero_division=0)
        if f1 > best_f1: best_f1, best_state, pc = f1, copy.deepcopy(model.state_dict()), 0
        else:
            pc += 1
            if pc >= 12: break

    model.load_state_dict(best_state)
    return model, X_tr, X_te, train_idx, test_idx

print("Training S13 LOSO fold for attention analysis...")
t0 = time.time()
model, X_tr_norm, X_te_norm, train_idx, test_idx = train_s13_fold()
print(f"Done in {time.time()-t0:.1f}s")

# =========================
# Pick the same 4 cases as SHAP
# =========================
s13_preds = preds_df[preds_df["subject"] == 13]
def pick_one(df):
    if len(df) == 0: return None
    return int(df.iloc[0]["window"])

cases = {
    "correct_fall":   pick_one(s13_preds[(s13_preds.y_true==1) & (s13_preds.y_pred==1)]),
    "missed_fall":    pick_one(s13_preds[(s13_preds.y_true==1) & (s13_preds.y_pred==0)]),
    "false_positive": pick_one(s13_preds[(s13_preds.y_true==0) & (s13_preds.y_pred==1)]),
    "correct_reject": pick_one(s13_preds[(s13_preds.y_true==0) & (s13_preds.y_pred==0)]),
}
global_to_local = {int(g): i for i, g in enumerate(test_idx)}

# =========================
# Collect attention weights
# =========================
model.eval()
attention_data = {}
for tag, global_idx in cases.items():
    if global_idx is None: continue
    local_idx = global_to_local[global_idx]
    x = torch.tensor(X_te_norm[local_idx:local_idx+1], dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        _ = model(x)
        attn = model.last_attn.cpu().numpy().squeeze()
    attention_data[tag] = {
        "global_idx": global_idx,
        "attention": attn,
        "signal": X[global_idx],
    }

# =========================
# Plot attention over signal
# =========================
fig, axes = plt.subplots(4, 1, figsize=(14, 10))
colors = {"correct_fall":"blue", "missed_fall":"orange",
          "false_positive":"red", "correct_reject":"green"}

def accel_mag(w):
    return np.sqrt(w[:, 12]**2 + w[:, 13]**2 + w[:, 14]**2)

for ax, (tag, d) in zip(axes, attention_data.items()):
    mag = accel_mag(d["signal"])
    attn = d["attention"]

    ax.plot(mag, color=colors[tag], linewidth=1.8, label="waist accel magnitude")
    ax.set_ylabel("|acc|")
    ax.grid(alpha=0.3)

    ax2 = ax.twinx()
    ax2.fill_between(range(100), 0, attn, alpha=0.35, color="purple", label="attention")
    ax2.set_ylabel("attention weight", color="purple")
    ax2.tick_params(axis="y", labelcolor="purple")
    ax2.set_ylim(0, max(attn.max() * 1.1, 0.05))

    row = s13_preds[s13_preds.window == d["global_idx"]].iloc[0]
    act = ACTIVITY_NAMES.get(int(activities[d["global_idx"]]), "?")
    ax.set_title(f"{tag}: w{d['global_idx']}  activity={act}  "
                 f"true={int(row.y_true)} pred={int(row.y_pred)} prob={row.prob:.2f}",
                 fontsize=10)

axes[-1].set_xlabel("Time step")
plt.suptitle("Temporal Attention — LOSO model, Subject 13", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "attention_loso_s13.png"), dpi=120)
plt.show()

# Save attention arrays
np.savez(os.path.join(OUTPUT_DIR, "attention_loso_s13.npz"),
         **{f"{tag}_attn": d["attention"] for tag, d in attention_data.items()},
         **{f"{tag}_global_idx": np.array(d["global_idx"])
            for tag, d in attention_data.items()})
print(f"\nSaved figure + attention data → {OUTPUT_DIR}")