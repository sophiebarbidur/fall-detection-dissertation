# =========================
# Step 3: Per-subject normalization — does it help LOSO?
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
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(PROJECT_ROOT, "prepared_upfall")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "outputs", "lstm")
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED); torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(RANDOM_SEED)

# Load
X      = np.load(os.path.join(DATA_DIR, "X_windows.npy"))
y      = np.load(os.path.join(DATA_DIR, "y.npy"))
groups = np.load(os.path.join(DATA_DIR, "groups.npy"))
with open(os.path.join(DATA_DIR, "loso_folds.json")) as f:
    folds = json.load(f)

# Hyperparams (exact match to original)
BATCH_SIZE, LR, WEIGHT_DECAY, DROPOUT = 64, 1e-3, 1e-4, 0.3
MAX_EPOCHS, PATIENCE, MAX_POS_WEIGHT, VAL_FRAC = 100, 12, 3.0, 0.15
HIDDEN_SIZE, NUM_LAYERS, BIDIRECTIONAL = 128, 2, True

def augment_batch(X):
    X = X.clone()
    B = X.shape[0]
    X = X + torch.randn_like(X) * 0.02
    scales = 0.9 + 0.2 * torch.rand(B, 1, 1, device=X.device)
    X = X * scales
    shifts = torch.randint(-5, 6, (B,), device=X.device).tolist()
    return torch.stack([torch.roll(X[i], shifts[i], dims=0) for i in range(B)])

class FallDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

class TemporalAttention(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.attn = nn.Linear(d, 1)
    def forward(self, x):
        w = F.softmax(self.attn(x).squeeze(-1), dim=1).unsqueeze(-1)
        return (x * w).sum(dim=1)

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
    def forward(self, x):
        h, _ = self.lstm(x); return self.head(self.drop(self.att(self.ln(h))))

# =========================
# KEY CHANGE: per-subject normalization
# =========================
def per_subject_normalise(X, groups):
    """
    Z-score each subject independently using their own mean/std across all their windows.
    Removes inter-subject amplitude differences.
    """
    X_norm = np.empty_like(X)
    for subj in np.unique(groups):
        mask = groups == subj
        flat = X[mask].reshape(-1, X.shape[2])         # (n_windows * T, C)
        mu   = flat.mean(axis=0)                        # (C,)
        sd   = flat.std(axis=0) + 1e-8                  # (C,)
        X_norm[mask] = (X[mask] - mu) / sd
    return X_norm

print("Applying per-subject normalization...")
X_psn = per_subject_normalise(X, groups)
print(f"Per-subject normalized shape: {X_psn.shape}")
print(f"  Overall mean: {X_psn.mean():.4f} (should be ~0)")
print(f"  Overall std : {X_psn.std():.4f} (should be ~1)")

# =========================
# LOSO with per-subject normalised data
# =========================
def train_fold(X_tr, y_tr, X_te, y_te):
    sss = StratifiedShuffleSplit(1, test_size=VAL_FRAC, random_state=RANDOM_SEED)
    tr, val = next(sss.split(X_tr, y_tr))
    Xa, ya, Xv, yv = X_tr[tr], y_tr[tr], X_tr[val], y_tr[val]

    n_pos = max(int(np.sum(ya == 1)), 1)
    pw = torch.tensor([min(int(np.sum(ya == 0)) / n_pos, MAX_POS_WEIGHT)],
                      dtype=torch.float32).to(DEVICE)

    tl = DataLoader(FallDataset(Xa, ya), batch_size=BATCH_SIZE, shuffle=True)
    vl = DataLoader(FallDataset(Xv, yv), batch_size=BATCH_SIZE)
    el = DataLoader(FallDataset(X_te, y_te), batch_size=BATCH_SIZE)

    model = FallLSTM(X_tr.shape[2]).to(DEVICE)
    crit  = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch   = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=4)

    best_f1, best_state, pc = -1, None, 0
    for ep in range(MAX_EPOCHS):
        model.train()
        for xb, yb in tl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE); xb = augment_batch(xb)
            opt.zero_grad(); crit(model(xb), yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval(); vp = []
        with torch.no_grad():
            for xb,_ in vl: vp.extend(torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy().flatten())
        f1 = f1_score(yv, (np.array(vp) > 0.5).astype(int), zero_division=0)
        sch.step(f1)
        if f1 > best_f1: best_f1, best_state, pc = f1, copy.deepcopy(model.state_dict()), 0
        else:
            pc += 1
            if pc >= PATIENCE: break

    model.load_state_dict(best_state)
    model.eval(); vp = []
    with torch.no_grad():
        for xb,_ in vl: vp.extend(torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy().flatten())
    vp = np.array(vp); best_t, bf = 0.5, 0.0
    for t in np.arange(0.20, 0.81, 0.05):
        f = f1_score(yv, (vp > t).astype(int), zero_division=0)
        if f > bf: bf, best_t = f, float(t)

    tp = []
    with torch.no_grad():
        for xb,_ in el: tp.extend(torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy().flatten())
    return (np.array(tp) > best_t).astype(int), best_t

fold_results = []
all_true, all_pred = [], []
t0 = time.time()
print(f"\n{'='*60}\nLOSO with PER-SUBJECT NORMALIZATION\n{'='*60}\n")

for i, fold in enumerate(folds, 1):
    subj      = fold["test_subject"]
    train_idx = np.array(fold["train_indices"])
    test_idx  = np.array(fold["test_indices"])

    # Use per-subject normalised data (already done globally)
    X_tr = X_psn[train_idx]; X_te = X_psn[test_idx]
    y_tr = y[train_idx];     y_te = y[test_idx]

    t1 = time.time()
    preds, thr = train_fold(X_tr, y_tr, X_te, y_te)

    acc  = accuracy_score(y_te, preds)
    prec = precision_score(y_te, preds, zero_division=0)
    rec  = recall_score(y_te, preds, zero_division=0)
    f1   = f1_score(y_te, preds, zero_division=0)
    print(f"Fold {i:>2}/15  Subj{subj:02d}  acc={acc:.2%}  f1={f1:.3f}  "
          f"prec={prec:.3f}  rec={rec:.3f}  ({time.time()-t1:.1f}s)")

    fold_results.append({"test_subject": int(subj), "accuracy": float(acc),
                         "precision": float(prec), "recall": float(rec),
                         "f1": float(f1), "threshold": float(thr)})
    all_true.extend(y_te.tolist()); all_pred.extend(preds.tolist())

total = time.time() - t0
accs = [r["accuracy"] for r in fold_results]
f1s  = [r["f1"]       for r in fold_results]

# =========================
# Compare to baseline
# =========================
baseline = json.load(open(os.path.join(OUTPUT_DIR, "lstm_5.6s_results.json")))

print(f"\n{'='*60}\nCOMPARISON\n{'='*60}")
print(f"{'Subject':>8}  {'Baseline F1':>12}  {'Per-subj F1':>12}  {'Δ':>7}")
print("-" * 45)
bl_by_subj = {r["test_subject"]: r["f1"] for r in baseline["fold_results"]}
for r in fold_results:
    s = r["test_subject"]; new = r["f1"]; old = bl_by_subj.get(s, 0)
    delta = new - old
    marker = "↑" if delta > 0.02 else ("↓" if delta < -0.02 else " ")
    print(f"S{s:>6d}  {old:>12.3f}  {new:>12.3f}  {delta:>+7.3f} {marker}")
print("-" * 45)
print(f"{'Mean':>8}  {baseline['mean_f1']:>12.3f}  {np.mean(f1s):>12.3f}  "
      f"{np.mean(f1s) - baseline['mean_f1']:>+7.3f}")
print(f"{'Std':>8}  {baseline['std_f1']:>12.3f}  {np.std(f1s):>12.3f}")
print(f"\nTotal time: {total:.1f}s")

results = {"model": "BiLSTM-Attention", "evaluation": "LOSO",
           "normalization": "per_subject",
           "mean_accuracy": float(np.mean(accs)), "std_accuracy": float(np.std(accs)),
           "mean_f1": float(np.mean(f1s)), "std_f1": float(np.std(f1s)),
           "pooled_accuracy": float(accuracy_score(all_true, all_pred)),
           "pooled_f1": float(f1_score(all_true, all_pred, zero_division=0)),
           "pooled_confusion_matrix": confusion_matrix(all_true, all_pred).tolist(),
           "fold_results": fold_results}

with open(os.path.join(OUTPUT_DIR, "lstm_per_subject_norm_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved → {os.path.join(OUTPUT_DIR, 'lstm_per_subject_norm_results.json')}")