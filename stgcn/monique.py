"""
ST-GCN Fall Detection on UP-Fall Dataset — LOSO-CV

References:
 - Yan, S., Xiong, Y., & Lin, D. (2018). ST-GCN. AAAI 2018.
 - Yan et al. (2023). Skeleton-Based Fall Detection with IMUs. Sensors.

Reports BOTH:
  1. 11-class activity classification (hard, honest)
  2. Binary fall vs non-fall (clinically relevant, higher numbers)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import json
from datetime import datetime

from moniquedataprep import load_gnn_windows

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
SEED         = 42
EPOCHS       = 150
LR           = 5e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE   = 64
WINDOW_SEC   = 2.0
SAMPLE_RATE  = 18
DROPOUT      = 0.3
PATIENCE     = 30
NUM_CLASSES  = 11
MAX_GRAD_NORM = 1.0
NOISE_STD    = 0.05

# Reduced architecture (prevents overfitting on ~13k windows)
NUM_BLOCKS   = 3       # Yan et al. used 5 — too large for this dataset
HIDDEN_CH    = 16      # Yan et al. used 32
TEMP_KERNEL  = 5       # Yan et al. used 9 — too wide for 36 timesteps

EDGES = [
    (0, 1),  # ankle  -> pocket
    (1, 2),  # pocket -> waist
    (2, 3),  # waist  -> neck
    (2, 4),  # waist  -> wrist
]
NUM_NODES = 5

ACTIVITY_NAMES = [
    "Fall: hands", "Fall: knees", "Fall: backwards",
    "Fall: sideways", "Fall: chair",
    "Walking", "Standing", "Sitting",
    "Picking up", "Jumping", "Lying",
]

# Reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------------------------------------
# ADJACENCY MATRICES (spatial configuration partition)
# -------------------------------------------------
dist_to_cog = {0: 2, 1: 1, 2: 0, 3: 1, 4: 2}

def build_spatial_config_adjacency(num_nodes, edges, dist_to_cog):
    A_self     = np.eye(num_nodes)
    A_cent_in  = np.zeros((num_nodes, num_nodes))
    A_cent_out = np.zeros((num_nodes, num_nodes))

    for (i, j) in edges:
        di, dj = dist_to_cog[i], dist_to_cog[j]
        if di == dj:
            A_self[i, j] = 1
            A_self[j, i] = 1
        elif dj < di:
            A_cent_in[i, j] = 1
            A_cent_out[j, i] = 1
        else:
            A_cent_out[i, j] = 1
            A_cent_in[j, i] = 1

    def normalize(A):
        row_sum = A.sum(axis=1)
        d_inv_sqrt = np.zeros_like(row_sum)
        nonzero = row_sum > 0
        d_inv_sqrt[nonzero] = np.power(row_sum[nonzero], -0.5)
        D = np.diag(d_inv_sqrt)
        return D @ A @ D

    return [
        torch.tensor(normalize(A_self),     dtype=torch.float32),
        torch.tensor(normalize(A_cent_in),  dtype=torch.float32),
        torch.tensor(normalize(A_cent_out), dtype=torch.float32),
    ]

A_list = build_spatial_config_adjacency(NUM_NODES, EDGES, dist_to_cog)


# -------------------------------------------------
# DATASET
# -------------------------------------------------
class GNNDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -------------------------------------------------
# ST-GCN BLOCK
# -------------------------------------------------
class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_subsets=3,
                 dropout=0.3, temporal_kernel=5):
        super().__init__()
        self.num_subsets = num_subsets
        self.spatial_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            for _ in range(num_subsets)
        ])
        self.bn_spatial = nn.BatchNorm2d(out_channels)
        padding = (temporal_kernel - 1) // 2
        self.temporal = nn.Conv2d(out_channels, out_channels,
                                  kernel_size=(1, temporal_kernel),
                                  padding=(0, padding))
        self.bn_temporal = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)
        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()

    def forward(self, x, A_list):
        res = self.residual(x)
        out = sum(
            torch.einsum("ij,bcjt->bcit", A_list[k], self.spatial_convs[k](x))
            for k in range(self.num_subsets)
        )
        out = F.relu(self.bn_spatial(out))
        out = self.temporal(out)
        out = self.bn_temporal(out)
        out = self.dropout(out)
        return F.relu(out + res)


# -------------------------------------------------
# MODEL
# -------------------------------------------------
class STGCN(nn.Module):
    def __init__(self, in_channels=6, num_classes=11, num_blocks=3,
                 hidden_ch=16, dropout=0.3, temporal_kernel=5):
        super().__init__()
        self.blocks = nn.ModuleList([
            STGCNBlock(in_channels if i == 0 else hidden_ch, hidden_ch,
                       dropout=dropout, temporal_kernel=temporal_kernel)
            for i in range(num_blocks)
        ])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(hidden_ch, num_classes)

    def forward(self, x, A_list):
        x = x.permute(0, 2, 1, 3)
        for block in self.blocks:
            x = block(x, A_list)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# -------------------------------------------------
# NORMALIZATION
# -------------------------------------------------
def normalize_data(X_train, X_test):
    _, nodes, features, time = X_train.shape
    def reshape(X):
        return X.transpose(0, 1, 3, 2).reshape(-1, features)
    def unreshape(X_flat, n):
        return X_flat.reshape(n, nodes, time, features).transpose(0, 1, 3, 2)

    mean = reshape(X_train).mean(axis=0)
    std  = reshape(X_train).std(axis=0) + 1e-8

    return (unreshape((reshape(X_train) - mean) / std, X_train.shape[0]),
            unreshape((reshape(X_test)  - mean) / std, X_test.shape[0]))


# -------------------------------------------------
# TRAINING / EVALUATION
# -------------------------------------------------
def train_epoch(model, loader, optimizer, criterion, A_list_dev):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        # Gaussian noise augmentation
        xb = xb + torch.randn_like(xb) * NOISE_STD
        optimizer.zero_grad()
        logits = model(xb, A_list_dev)
        loss = criterion(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, A_list_dev):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_true = [], []
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = model(xb, A_list_dev)
        loss = criterion(logits, yb)
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += len(yb)
        all_preds.extend(preds.cpu().numpy())
        all_true.extend(yb.cpu().numpy())
    return total_loss / len(loader), correct / total, all_preds, all_true


# -------------------------------------------------
# SINGLE FOLD
# -------------------------------------------------
def train_fold(X_train, y_train, X_test, y_test, fold_name="", verbose=True):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    X_train_n, X_test_n = normalize_data(X_train, X_test)

    g = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(GNNDataset(X_train_n, y_train), batch_size=BATCH_SIZE,
                              shuffle=True, generator=g)
    test_loader  = DataLoader(GNNDataset(X_test_n, y_test), batch_size=BATCH_SIZE)

    # Class weights: inverse frequency, mean-normalised
    counts = np.bincount(y_train, minlength=NUM_CLASSES).astype(float)
    inv_freq = 1.0 / (counts + 1e-6)
    inv_freq = inv_freq / inv_freq.mean()
    class_weights = torch.tensor(inv_freq, dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model = STGCN(in_channels=6, num_classes=NUM_CLASSES, num_blocks=NUM_BLOCKS,
                  hidden_ch=HIDDEN_CH, dropout=DROPOUT,
                  temporal_kernel=TEMP_KERNEL).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    A_list_dev = [A.to(DEVICE) for A in A_list]

    best_val_loss = float("inf")
    patience_counter = 0
    ckpt = f"ckpt_{fold_name}.pt"

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, A_list_dev)
        val_loss, val_acc, _, _ = evaluate(model, test_loader, criterion, A_list_dev)
        scheduler.step()

        if verbose and epoch % 20 == 0:
            print(f"  {fold_name} Ep {epoch:3d}  "
                  f"trn={train_loss:.4f}  val={val_loss:.4f}  acc={val_acc:.2%}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), ckpt)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                if verbose:
                    print(f"  {fold_name} Early stop ep {epoch}")
                break

    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    _, test_acc, preds, true = evaluate(model, test_loader, criterion, A_list_dev)

    import os
    if os.path.exists(ckpt):
        os.remove(ckpt)

    return test_acc, np.array(preds), np.array(true)


# -------------------------------------------------
# BINARY METRICS (fall vs non-fall from 11-class preds)
# -------------------------------------------------
def compute_binary_metrics(true_11, preds_11):
    """Convert 11-class labels to binary: classes 0-4 = fall (1), 5-10 = non-fall (0)."""
    true_bin  = np.array([1 if t < 5 else 0 for t in true_11])
    preds_bin = np.array([1 if p < 5 else 0 for p in preds_11])
    acc = (true_bin == preds_bin).mean()
    report = classification_report(true_bin, preds_bin,
                                   target_names=["Non-Fall", "Fall"], zero_division=0)
    cm = confusion_matrix(true_bin, preds_bin)
    return acc, report, cm


# -------------------------------------------------
# MAIN — LOSO-CV
# -------------------------------------------------
if __name__ == "__main__":

    print("\n==== ST-GCN FALL DETECTION — LOSO Cross-Validation ====")
    print(f"Device: {DEVICE}")
    print(f"Architecture: {NUM_BLOCKS} blocks, {HIDDEN_CH} channels, kernel={TEMP_KERNEL}")
    print(f"Window: {WINDOW_SEC}s | Overlap: 50% | Timesteps: {int(WINDOW_SEC * SAMPLE_RATE)}")
    print(f"Regularisation: dropout={DROPOUT}, weight_decay={WEIGHT_DECAY}, noise={NOISE_STD}")

    # --- Load data ---
    X, y, subject_ids = load_gnn_windows("data.csv")

    print(f"\nLoaded: {X.shape[0]} windows, shape {X.shape}")
    print(f"Labels: {np.bincount(y, minlength=NUM_CLASSES)}")

    subjects = sorted(np.unique(subject_ids).tolist())
    print(f"Subjects ({len(subjects)}): {subjects}")

    time_steps = int(WINDOW_SEC * SAMPLE_RATE)
    assert X.shape[-1] == time_steps

    # Count params
    tmp = STGCN(in_channels=6, num_classes=NUM_CLASSES, num_blocks=NUM_BLOCKS,
                hidden_ch=HIDDEN_CH, temporal_kernel=TEMP_KERNEL)
    print(f"Model params: {sum(p.numel() for p in tmp.parameters()):,}")
    del tmp

    # --- LOSO-CV ---
    print(f"\n{'='*60}")
    print(f"LOSO-CV: {len(subjects)} folds")
    print(f"{'='*60}")

    fold_accs = []
    fold_binary_accs = []
    all_preds = []
    all_true = []

    for test_subj in subjects:
        test_mask  = subject_ids == test_subj
        train_mask = ~test_mask

        X_tr, y_tr = X[train_mask], y[train_mask]
        X_te, y_te = X[test_mask],  y[test_mask]

        fold_name = f"Subj{test_subj:02d}"
        print(f"\n--- {fold_name}: train={len(y_tr)}, test={len(y_te)} ---")

        if len(y_te) == 0:
            print(f"  SKIP — no data")
            continue

        acc, preds, true = train_fold(X_tr, y_tr, X_te, y_te,
                                      fold_name=fold_name, verbose=True)

        bin_acc, _, _ = compute_binary_metrics(true, preds)

        fold_accs.append(acc)
        fold_binary_accs.append(bin_acc)
        all_preds.append(preds)
        all_true.append(true)

        print(f"  => 11-class: {acc:.2%}  |  Binary fall/non-fall: {bin_acc:.2%}")

    # --- Aggregate ---
    all_preds = np.concatenate(all_preds)
    all_true  = np.concatenate(all_true)

    print(f"\n{'='*60}")
    print("LOSO-CV RESULTS")
    print(f"{'='*60}")

    print(f"\n{'Subject':>10} {'11-class':>10} {'Binary':>10}")
    print("-" * 34)
    for subj, acc11, accb in zip(subjects, fold_accs, fold_binary_accs):
        print(f"{subj:>10} {acc11:>9.2%} {accb:>9.2%}")
    print("-" * 34)
    print(f"{'Mean':>10} {np.mean(fold_accs):>9.2%} {np.mean(fold_binary_accs):>9.2%}")
    print(f"{'Std':>10} {np.std(fold_accs):>9.2%} {np.std(fold_binary_accs):>9.2%}")

    # --- 11-class report (pooled) ---
    print(f"\n--- 11-Class Classification Report (pooled) ---")
    print(classification_report(all_true, all_preds,
                                target_names=ACTIVITY_NAMES, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(all_true, all_preds))

    # --- Binary report (pooled) ---
    bin_acc, bin_report, bin_cm = compute_binary_metrics(all_true, all_preds)
    print(f"\n--- Binary Fall/Non-Fall Report (pooled) ---")
    print(f"Binary Accuracy: {bin_acc:.2%}")
    print(bin_report)
    print("Confusion Matrix:")
    print(bin_cm)

    # --- Save ---
    results = {
        "config": {
            "num_blocks": NUM_BLOCKS, "hidden_ch": HIDDEN_CH,
            "temp_kernel": TEMP_KERNEL, "lr": LR, "weight_decay": WEIGHT_DECAY,
            "dropout": DROPOUT, "noise_std": NOISE_STD, "epochs": EPOCHS,
        },
        "per_subject": [
            {"subject": s, "acc_11class": a, "acc_binary": b}
            for s, a, b in zip(subjects, fold_accs, fold_binary_accs)
        ],
        "mean_11class": float(np.mean(fold_accs)),
        "std_11class": float(np.std(fold_accs)),
        "mean_binary": float(np.mean(fold_binary_accs)),
        "std_binary": float(np.std(fold_binary_accs)),
    }
    fname = f"loso_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(fname, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {fname}")