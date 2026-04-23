# =========================
# 0) Mount Drive + GPU check
# =========================


import os, json, time, copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))

# =========================
# 1) Paths
# =========================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(PROJECT_ROOT, "prepared_upfall")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "outputs", "cnn")
os.makedirs(OUTPUT_DIR, exist_ok=True)

assert os.path.exists(DATA_DIR), f"DATA_DIR not found: {DATA_DIR}"
print("DATA_DIR contents:", os.listdir(DATA_DIR))

# =========================
# 2) Repro + device
# =========================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE =", DEVICE)

# =========================
# 3) Hyperparameters (aligned with LSTM for fair comparison)
# =========================
MAX_EPOCHS     = 100
PATIENCE       = 12
BATCH_SIZE     = 64
LR             = 1e-3
WEIGHT_DECAY   = 1e-4
DROPOUT        = 0.3
VAL_FRAC       = 0.15
MAX_POS_WEIGHT = 3.0

NUM_WORKERS = 2
PIN_MEMORY  = torch.cuda.is_available()

# =========================
# 4) Dataset (Conv1d wants NxCxT, not NxTxC)
# =========================
class FallDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)  # (N,C,T)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

# =========================
# 5) Model — 1D-CNN with residual blocks
# =========================
class ResBlock1D(nn.Module):
    """Conv -> BN -> ReLU -> Conv -> BN  +  skip."""
    def __init__(self, channels, kernel_size):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.bn1   = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.bn2   = nn.BatchNorm1d(channels)
        self.relu  = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + x)

class FallCNN(nn.Module):
    def __init__(self, in_channels: int, dropout: float = DROPOUT):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.stage1 = nn.Sequential(
            ResBlock1D(64, 5),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )
        self.expand = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.stage2 = nn.Sequential(
            ResBlock1D(128, 3),
            nn.Dropout(dropout),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.expand(x)
        x = self.stage2(x)
        x = self.gap(x).squeeze(-1)
        return self.head(x)

# =========================
# 6) Helpers
# =========================
def normalise_per_channel(X_train, X_test):
    X_tr = np.empty_like(X_train)
    X_te = np.empty_like(X_test)
    for c in range(X_train.shape[2]):
        sc = StandardScaler()
        X_tr[:, :, c] = sc.fit_transform(X_train[:, :, c])
        X_te[:, :, c] = sc.transform(X_test[:, :, c])
    return X_tr, X_te

def make_val_split(X, y):
    sss = StratifiedShuffleSplit(1, test_size=VAL_FRAC, random_state=RANDOM_SEED)
    tr, val = next(sss.split(X, y))
    return X[tr], y[tr], X[val], y[val]

def compute_pos_weight(y_train):
    n_pos = max(int(np.sum(y_train == 1)), 1)
    raw   = int(np.sum(y_train == 0)) / n_pos
    w     = min(raw, MAX_POS_WEIGHT)
    print(f"  pos_weight raw={raw:.2f} → capped={w:.2f}", end="")
    return torch.tensor([w], dtype=torch.float32).to(DEVICE)

def tune_threshold(model, loader, y_true):
    model.eval()
    probs = []
    with torch.no_grad():
        for xb, _ in loader:
            probs.extend(torch.sigmoid(model(xb.to(DEVICE, non_blocking=True))).cpu().numpy().flatten())
    probs = np.array(probs)
    best_t, best_f = 0.5, 0.0
    for t in np.arange(0.20, 0.81, 0.05):
        f = f1_score(y_true, (probs > t).astype(int), zero_division=0)
        if f > best_f:
            best_f, best_t = f, float(t)
    return best_t

# =========================
# 7) Train fold
# =========================
def train_fold(X_train_norm, y_train, X_test_norm, y_test):
    X_tr, y_tr, X_val, y_val = make_val_split(X_train_norm, y_train)
    pos_weight = compute_pos_weight(y_tr)

    train_loader = DataLoader(FallDataset(X_tr, y_tr), batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader   = DataLoader(FallDataset(X_val, y_val), batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader  = DataLoader(FallDataset(X_test_norm, y_test), batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    model     = FallCNN(in_channels=X_tr.shape[2]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimiser = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="max", factor=0.5, patience=4, min_lr=1e-6)

    best_val_f1, best_state, best_epoch, patience_ctr = -1.0, None, 0, 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            optimiser.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

        model.eval()
        val_probs = []
        with torch.no_grad():
            for xb, _ in val_loader:
                val_probs.extend(torch.sigmoid(model(xb.to(DEVICE, non_blocking=True))).cpu().numpy().flatten())
        val_f1 = f1_score(y_val, (np.array(val_probs) > 0.5).astype(int), zero_division=0)
        scheduler.step(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1  = val_f1
            best_state   = copy.deepcopy(model.state_dict())
            best_epoch   = epoch
            patience_ctr = 0
        else:
            patience_ctr += 1
        if patience_ctr >= PATIENCE:
            break

    model.load_state_dict(best_state)
    threshold = tune_threshold(model, val_loader, y_val)

    model.eval()
    test_probs = []
    with torch.no_grad():
        for xb, _ in test_loader:
            test_probs.extend(torch.sigmoid(model(xb.to(DEVICE, non_blocking=True))).cpu().numpy().flatten())
    preds = (np.array(test_probs) > threshold).astype(int)
    return preds, best_epoch, threshold

# =========================
# 8) Main
# =========================
print("Loading dataset...")
X = np.load(os.path.join(DATA_DIR, "X_windows.npy"))
y = np.load(os.path.join(DATA_DIR, "y.npy"))
with open(os.path.join(DATA_DIR, "loso_folds.json")) as f:
    folds = json.load(f)

N, T, C = X.shape
print(f"X shape: {X.shape}")
print(f"Falls: {np.sum(y==1)}, Non-falls: {np.sum(y==0)}")
print(f"LOSO folds: {len(folds)}")

assert T == 100, f"ERROR: Expected 100 timesteps (5.6s windows), got {T}. Wrong data!"

fold_results = []
all_y_true, all_y_pred = [], []
t_total = time.time()

print(f"\n{'='*60}\nLOSO-CV — 1D-CNN (5.6s windows)\n{'='*60}\n")

for i, fold in enumerate(folds, start=1):
    subj      = fold["test_subject"]
    train_idx = np.array(fold["train_indices"])
    test_idx  = np.array(fold["test_indices"])

    X_tr_norm, X_te_norm = normalise_per_channel(X[train_idx], X[test_idx])
    y_tr, y_te = y[train_idx], y[test_idx]

    t0 = time.time()
    print(f"Fold {i:>2}/{len(folds)} Subj{subj:02d} train={len(train_idx)} test={len(test_idx)} falls={np.sum(y_te==1)}")

    preds, best_ep, thr = train_fold(X_tr_norm, y_tr, X_te_norm, y_te)

    acc  = accuracy_score(y_te, preds)
    prec = precision_score(y_te, preds, zero_division=0)
    rec  = recall_score(y_te, preds, zero_division=0)
    f1   = f1_score(y_te, preds, zero_division=0)
    dt   = time.time() - t0

    print(f"  → acc={acc:.2%} f1={f1:.3f} prec={prec:.3f} rec={rec:.3f} thr={thr:.2f} ep={best_ep} ({dt:.1f}s)\n")

    fold_results.append({"test_subject": int(subj), "accuracy": float(acc),
        "precision": float(prec), "recall": float(rec), "f1": float(f1),
        "best_epoch": int(best_ep), "threshold": float(thr),
        "num_test": int(len(test_idx)), "num_falls": int(np.sum(y_te==1)),
        "time_s": round(dt, 1)})
    all_y_true.extend(y_te.tolist())
    all_y_pred.extend(preds.tolist())

total_time = time.time() - t_total
all_y_true = np.array(all_y_true)
all_y_pred = np.array(all_y_pred)

accs = [r["accuracy"] for r in fold_results]
f1s  = [r["f1"]       for r in fold_results]

print(f"\n{'='*60}\nRESULTS — 1D-CNN (5.6s windows)\n{'='*60}")
print(f"\n{'Subj':>5}  {'Acc':>8}  {'F1':>7}  {'Prec':>7}  {'Rec':>7}  {'Thr':>5}  Ep")
print("-" * 60)
for r in fold_results:
    print(f"{r['test_subject']:>5}  {r['accuracy']:>7.2%}  {r['f1']:>7.3f}"
          f"  {r['precision']:>7.3f}  {r['recall']:>7.3f}"
          f"  {r['threshold']:>5.2f}  {r['best_epoch']}")
print("-" * 60)
print(f"{'Mean':>5}  {np.mean(accs):>7.2%}  {np.mean(f1s):>7.3f}")
print(f"{'Std':>5}  {np.std(accs):>7.2%}  {np.std(f1s):>7.3f}")

print("\n--- Pooled classification report ---")
print(classification_report(all_y_true, all_y_pred, target_names=["Non-Fall","Fall"], zero_division=0))
print("--- Pooled confusion matrix (rows=true, cols=pred) ---")
print(confusion_matrix(all_y_true, all_y_pred))
print(f"\nTotal time: {total_time:.1f}s")

results = {"model": "1D-CNN-Residual", "evaluation": "LOSO", "data": "5.6s_windows",
    "hyperparameters": {"max_epochs": MAX_EPOCHS, "patience": PATIENCE,
        "batch_size": BATCH_SIZE, "lr": LR, "weight_decay": WEIGHT_DECAY,
        "dropout": DROPOUT, "max_pos_weight": MAX_POS_WEIGHT, "val_frac": VAL_FRAC},
    "num_folds": len(folds), "mean_accuracy": float(np.mean(accs)),
    "std_accuracy": float(np.std(accs)), "mean_f1": float(np.mean(f1s)),
    "std_f1": float(np.std(f1s)),
    "pooled_accuracy": float(accuracy_score(all_y_true, all_y_pred)),
    "pooled_f1": float(f1_score(all_y_true, all_y_pred, zero_division=0)),
    "pooled_confusion_matrix": confusion_matrix(all_y_true, all_y_pred).tolist(),
    "total_time_s": round(total_time, 1), "fold_results": fold_results,
    "data_shape": list(X.shape)}

out = os.path.join(OUTPUT_DIR, "cnn_5.6s_results.json")
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved → {out}")