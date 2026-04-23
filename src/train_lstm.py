# =========================
# 0) Mount Drive + GPU check
# =========================


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
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
   print("gpu:", torch.cuda.get_device_name(0))


# =========================
# 1) Paths (EDIT ONLY IF your folder name differs)
# =========================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(PROJECT_ROOT, "prepared_upfall")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "outputs", "lstm")
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
# 3) Hyperparameters
# =========================
MAX_EPOCHS     = 100
PATIENCE       = 12
BATCH_SIZE     = 64
LR             = 1e-3
WEIGHT_DECAY   = 1e-4
DROPOUT        = 0.3
VAL_FRAC       = 0.15
MAX_POS_WEIGHT = 3.0


HIDDEN_SIZE   = 128
NUM_LAYERS    = 2
BIDIRECTIONAL = True


# DataLoader perf (good defaults for GPU)
NUM_WORKERS = 2
PIN_MEMORY  = torch.cuda.is_available()


# =========================
# 4) Augmentation
# =========================
def augment_batch(X: torch.Tensor) -> torch.Tensor:
   X = X.clone()
   B = X.shape[0]
   X = X + torch.randn_like(X) * 0.02
   scales = 0.9 + 0.2 * torch.rand(B, 1, 1, device=X.device)
   X = X * scales
   shifts = torch.randint(-5, 6, (B,), device=X.device).tolist()
   X = torch.stack([torch.roll(X[i], shifts[i], dims=0) for i in range(B)])
   return X


# =========================
# 5) Dataset
# =========================
class FallDataset(Dataset):
   def __init__(self, X: np.ndarray, y: np.ndarray):
       self.X = torch.tensor(X, dtype=torch.float32)   # (N,T,C)
       self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)


   def __len__(self): return len(self.X)
   def __getitem__(self, i): return self.X[i], self.y[i]


# =========================
# 6) Model
# =========================
class TemporalAttention(nn.Module):
   def __init__(self, hidden_dim: int):
       super().__init__()
       self.attn = nn.Linear(hidden_dim, 1)


   def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
       scores  = self.attn(lstm_out).squeeze(-1)        # (B,T)
       weights = F.softmax(scores, dim=1).unsqueeze(-1) # (B,T,1)
       return (lstm_out * weights).sum(dim=1)           # (B,H)


class FallLSTM(nn.Module):
   def __init__(self, input_size: int,
                hidden_size: int = HIDDEN_SIZE,
                num_layers: int = NUM_LAYERS,
                bidirectional: bool = BIDIRECTIONAL,
                dropout: float = DROPOUT):
       super().__init__()
       self.num_dir = 2 if bidirectional else 1
       out_dim = hidden_size * self.num_dir


       self.lstm = nn.LSTM(
           input_size=input_size,
           hidden_size=hidden_size,
           num_layers=num_layers,
           batch_first=True,
           dropout=dropout if num_layers > 1 else 0.0,
           bidirectional=bidirectional,
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


   def forward(self, x: torch.Tensor) -> torch.Tensor:
       out, _ = self.lstm(x)          # (B,T,out_dim)
       out    = self.layer_norm(out)
       pooled = self.attention(out)   # (B,out_dim)
       pooled = self.dropout(pooled)
       return self.head(pooled)


# =========================
# 7) Helpers
# =========================
def normalise_per_channel(X_train: np.ndarray, X_test: np.ndarray):
   X_tr = np.empty_like(X_train)
   X_te = np.empty_like(X_test)
   for c in range(X_train.shape[2]):
       sc = StandardScaler()
       X_tr[:, :, c] = sc.fit_transform(X_train[:, :, c])
       X_te[:, :, c] = sc.transform(X_test[:, :, c])
   return X_tr, X_te


def make_val_split(X: np.ndarray, y: np.ndarray):
   sss = StratifiedShuffleSplit(1, test_size=VAL_FRAC, random_state=RANDOM_SEED)
   tr, val = next(sss.split(X, y))
   return X[tr], y[tr], X[val], y[val]


def compute_pos_weight(y_train: np.ndarray) -> torch.Tensor:
   n_pos = max(int(np.sum(y_train == 1)), 1)
   n_neg = int(np.sum(y_train == 0))
   raw   = n_neg / n_pos
   w     = min(raw, MAX_POS_WEIGHT)
   print(f"  pos_weight raw={raw:.2f} → capped={w:.2f}", end="")
   return torch.tensor([w], dtype=torch.float32).to(DEVICE)


def tune_threshold(model: nn.Module, loader: DataLoader, y_true: np.ndarray) -> float:
   model.eval()
   probs = []
   with torch.no_grad():
       for xb, _ in loader:
           xb = xb.to(DEVICE, non_blocking=True)
           probs.extend(torch.sigmoid(model(xb)).cpu().numpy().flatten())
   probs = np.array(probs)


   best_t, best_f = 0.5, 0.0
   for t in np.arange(0.20, 0.81, 0.05):
       f = f1_score(y_true, (probs > t).astype(int), zero_division=0)
       if f > best_f:
           best_f, best_t = f, float(t)
   return best_t


# =========================
# 8) Train fold
# =========================
def train_fold(X_train_norm: np.ndarray, y_train: np.ndarray,
              X_test_norm: np.ndarray,  y_test:  np.ndarray):
   X_tr, y_tr, X_val, y_val = make_val_split(X_train_norm, y_train)
   pos_weight = compute_pos_weight(y_tr)


   train_loader = DataLoader(
       FallDataset(X_tr, y_tr),
       batch_size=BATCH_SIZE, shuffle=True,
       num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
   )
   val_loader = DataLoader(
       FallDataset(X_val, y_val),
       batch_size=BATCH_SIZE, shuffle=False,
       num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
   )
   test_loader = DataLoader(
       FallDataset(X_test_norm, y_test),
       batch_size=BATCH_SIZE, shuffle=False,
       num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
   )


   model     = FallLSTM(input_size=X_tr.shape[2]).to(DEVICE)
   criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
   optimiser = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
   scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
       optimiser, mode="max", factor=0.5, patience=4, min_lr=1e-6
   )


   best_val_f1, best_state, best_epoch, patience_ctr = -1.0, None, 0, 0


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


       model.eval()
       val_probs = []
       with torch.no_grad():
           for xb, _ in val_loader:
               xb = xb.to(DEVICE, non_blocking=True)
               val_probs.extend(torch.sigmoid(model(xb)).cpu().numpy().flatten())


       val_f1 = f1_score(y_val, (np.array(val_probs) > 0.5).astype(int), zero_division=0)
       scheduler.step(val_f1)


       if val_f1 > best_val_f1:
           best_val_f1 = val_f1
           best_state  = copy.deepcopy(model.state_dict())
           best_epoch  = epoch
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
           xb = xb.to(DEVICE, non_blocking=True)
           test_probs.extend(torch.sigmoid(model(xb)).cpu().numpy().flatten())


   preds = (np.array(test_probs) > threshold).astype(int)
   return preds, best_epoch, threshold


# =========================
# 9) Main LOSO run
# =========================
print("Loading dataset...")
X = np.load(os.path.join(DATA_DIR, "X_windows.npy"))
y = np.load(os.path.join(DATA_DIR, "y.npy"))
with open(os.path.join(DATA_DIR, "loso_folds.json")) as f:
   folds = json.load(f)


N, T, C = X.shape
HZ = 18
print(f"X shape: {X.shape}")
print(f"Window length: {T} steps = {T/HZ:.1f}s at {HZ}Hz")
print(f"Falls: {np.sum(y==1)}, Non-falls: {np.sum(y==0)}")
print(f"LOSO folds: {len(folds)}")


# Set this to None for full run, or an int like 1 or 2 to test quickly.
RUN_ONLY_N_FOLDS = None


folds_to_run = folds if RUN_ONLY_N_FOLDS is None else folds[:RUN_ONLY_N_FOLDS]


fold_results = []
all_y_true, all_y_pred = [], []
t_total = time.time()


print("\n" + "="*60)
print("LOSO-CV  —  BiLSTM + Attention")
print("="*60 + "\n")


for i, fold in enumerate(folds_to_run, start=1):
   subj      = fold["test_subject"]
   train_idx = np.array(fold["train_indices"])
   test_idx  = np.array(fold["test_indices"])


   X_tr_norm, X_te_norm = normalise_per_channel(X[train_idx], X[test_idx])
   y_tr, y_te = y[train_idx], y[test_idx]


   t0 = time.time()
   print(f"Fold {i:>2}/{len(folds_to_run)} Subj{subj:02d} train={len(train_idx)} test={len(test_idx)} falls_test={np.sum(y_te==1)}")


   preds, best_ep, thr = train_fold(X_tr_norm, y_tr, X_te_norm, y_te)


   acc  = accuracy_score(y_te, preds)
   prec = precision_score(y_te, preds, zero_division=0)
   rec  = recall_score(y_te, preds, zero_division=0)
   f1   = f1_score(y_te, preds, zero_division=0)
   dt   = time.time() - t0


   print(f"  → acc={acc:.2%} f1={f1:.3f} prec={prec:.3f} rec={rec:.3f} thr={thr:.2f} ep={best_ep} ({dt:.1f}s)\n")


   fold_results.append({
       "test_subject": int(subj),
       "accuracy": float(acc),
       "precision": float(prec),
       "recall": float(rec),
       "f1": float(f1),
       "best_epoch": int(best_ep),
       "threshold": float(thr),
       "num_test": int(len(test_idx)),
       "num_falls_test": int(np.sum(y_te == 1)),
       "time_s": round(dt, 1),
   })
   all_y_true.extend(y_te.tolist())
   all_y_pred.extend(preds.tolist())


total_time = time.time() - t_total
print("Done. Total time (s):", round(total_time, 1))


# Save partial/full results
out_path = os.path.join(OUTPUT_DIR, "lstm_results_colab.json")
with open(out_path, "w") as f:
   json.dump({"fold_results": fold_results, "total_time_s": round(total_time,1)}, f, indent=2)


print("Saved:", out_path)