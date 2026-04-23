# =========================
# BiLSTM + Attention (Random Split) — Colab-ready + SHAP-ready
# =========================

from google.colab import drive
drive.mount('/content/drive')

import os, json, time, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

# -------------------------
# Paths
# -------------------------
PROJECT_ROOT = "/content/drive/MyDrive/Falling"
DATA_DIR     = os.path.join(PROJECT_ROOT, "prepared_upfall_random")  # adjust if needed
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "outputs", "lstm_random")
os.makedirs(OUTPUT_DIR, exist_ok=True)

assert os.path.exists(DATA_DIR), f"DATA_DIR not found: {DATA_DIR}"
print("DATA_DIR contents:", os.listdir(DATA_DIR))

# -------------------------
# Repro + device
# -------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE =", DEVICE)

# -------------------------
# Hyperparameters
# -------------------------
MAX_EPOCHS     = 100
PATIENCE       = 12
BATCH_SIZE     = 64
LR             = 1e-3
WEIGHT_DECAY   = 1e-4
DROPOUT        = 0.3
MAX_POS_WEIGHT = 3.0

HIDDEN_SIZE   = 128
NUM_LAYERS    = 2
BIDIRECTIONAL = True

TEST_FRAC = 0.20   # single split for XAI
VAL_FRAC  = 0.15

NUM_WORKERS = 2
PIN_MEMORY  = torch.cuda.is_available()

# -------------------------
# Augmentation
# -------------------------
def augment_batch(X: torch.Tensor) -> torch.Tensor:
    X = X.clone()
    B = X.shape[0]
    X = X + torch.randn_like(X) * 0.02
    scales = 0.9 + 0.2 * torch.rand(B, 1, 1, device=X.device)
    X = X * scales
    shifts = torch.randint(-5, 6, (B,), device=X.device).tolist()
    X = torch.stack([torch.roll(X[i], shifts[i], dims=0) for i in range(B)])
    return X

# -------------------------
# Dataset
# -------------------------
class FallDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)          # (N,T,C)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

# -------------------------
# Model
# -------------------------
class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)
    def forward(self, lstm_out):
        scores  = self.attn(lstm_out).squeeze(-1)        # (B,T)
        weights = F.softmax(scores, dim=1).unsqueeze(-1) # (B,T,1)
        return (lstm_out * weights).sum(dim=1)           # (B,H)

class FallLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=HIDDEN_SIZE,
                 num_layers=NUM_LAYERS, bidirectional=BIDIRECTIONAL,
                 dropout=DROPOUT):
        super().__init__()
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        self.layer_norm = nn.LayerNorm(out_dim)
        self.attention  = TemporalAttention(out_dim)
        self.dropout    = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(out_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)          # (B,T,H)
        out    = self.layer_norm(out)
        pooled = self.attention(out)   # (B,H)
        pooled = self.dropout(pooled)
        return self.head(pooled)       # (B,1) logits

# -------------------------
# Helpers
# -------------------------
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
    n_neg = int(np.sum(y_train == 0))
    raw   = n_neg / n_pos
    w     = min(raw, MAX_POS_WEIGHT)
    print(f"  pos_weight raw={raw:.2f} → capped={w:.2f}")
    return torch.tensor([w], dtype=torch.float32).to(DEVICE)

def tune_threshold_from_probs(probs, y_true):
    best_t, best_f = 0.5, 0.0
    for t in np.arange(0.20, 0.81, 0.05):
        f = f1_score(y_true, (probs > t).astype(int), zero_division=0)
        if f > best_f:
            best_f, best_t = f, float(t)
    return best_t

def predict_proba(model, X_np):
    model.eval()
    loader = DataLoader(FallDataset(X_np, np.zeros(len(X_np))), batch_size=256,
                        shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    probs = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(DEVICE, non_blocking=True)
            p = torch.sigmoid(model(xb)).detach().cpu().numpy().flatten()
            probs.append(p)
    return np.concatenate(probs)

# -------------------------
# Train on ONE random split (best for SHAP)
# -------------------------
print("Loading dataset...")
X = np.load(os.path.join(DATA_DIR, "X_windows.npy"))
y = np.load(os.path.join(DATA_DIR, "y.npy"))

N, T, C = X.shape
print("X shape:", X.shape, " y:", y.shape)
print("Falls:", int(np.sum(y==1)), "Non-falls:", int(np.sum(y==0)))

# Create one stratified split
sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=RANDOM_SEED)
train_idx, test_idx = next(sss.split(X, y))

X_tr_raw, y_tr = X[train_idx], y[train_idx]
X_te_raw, y_te = X[test_idx],  y[test_idx]

# Normalize per channel (fit on train only)
X_tr_norm, X_te_norm = normalise_per_channel(X_tr_raw, X_te_raw)

# Create internal val split from training only
X_tr, y_tr2, X_val, y_val = make_val_split(X_tr_norm, y_tr)

pos_weight = compute_pos_weight(y_tr2)

train_loader = DataLoader(FallDataset(X_tr, y_tr2), batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
val_loader   = DataLoader(FallDataset(X_val, y_val), batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

model     = FallLSTM(input_size=C).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimiser = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimiser, mode="max", factor=0.5, patience=4, min_lr=1e-6
)

best_state = None
best_val_f1 = -1.0
best_epoch = 0
patience_ctr = 0

t0_all = time.time()
for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    for xb, yb in train_loader:
        xb = xb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)
        xb = augment_batch(xb)

        optimiser.zero_grad(set_to_none=True)
        loss = criterion(model(xb), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()

    # val
    model.eval()
    val_probs = []
    with torch.no_grad():
        for xb, _ in val_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            val_probs.extend(torch.sigmoid(model(xb)).cpu().numpy().flatten())
    val_probs = np.array(val_probs)

    val_f1 = f1_score(y_val, (val_probs > 0.5).astype(int), zero_division=0)
    scheduler.step(val_f1)

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = epoch
        patience_ctr = 0
    else:
        patience_ctr += 1

    if epoch % 5 == 0 or epoch == 1:
        print(f"epoch {epoch:03d} val_f1@0.5={val_f1:.3f} best={best_val_f1:.3f}")

    if patience_ctr >= PATIENCE:
        break

model.load_state_dict(best_state)

# Tune threshold on val set (using current model)
val_probs = predict_proba(model, X_val)
threshold = tune_threshold_from_probs(val_probs, y_val)
print("Best epoch:", best_epoch, " tuned threshold:", threshold)

# Evaluate on test
test_probs = predict_proba(model, X_te_norm)
preds = (test_probs > threshold).astype(int)

acc  = accuracy_score(y_te, preds)
prec = precision_score(y_te, preds, zero_division=0)
rec  = recall_score(y_te, preds, zero_division=0)
f1   = f1_score(y_te, preds, zero_division=0)

print("\nTest metrics")
print("acc:", acc, "prec:", prec, "rec:", rec, "f1:", f1)
print("\nConfusion matrix:\n", confusion_matrix(y_te, preds))
print("\nClassification report:\n", classification_report(y_te, preds, target_names=["Non-Fall","Fall"], zero_division=0))
print("Total train time (s):", round(time.time() - t0_all, 1))

# Save model
model_path = os.path.join(OUTPUT_DIR, "bilstm_attention_randomsplit.pth")
torch.save(model.state_dict(), model_path)
print("Saved model:", model_path)

# Keep SHAP variables available:
# model, X_tr_norm, X_te_norm, y_te, preds, test_probs, threshold