# =========================
# Per-subject, per-fall-type error analysis
# =========================
import os, json, copy, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score

# =========================
# Paths
# =========================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR_RND = os.path.join(PROJECT_ROOT, "prepared_upfall")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "outputs", "xai")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Load data + original activity labels
# =========================
X       = np.load(os.path.join(DATA_DIR, "X_windows.npy"))
y       = np.load(os.path.join(DATA_DIR, "y.npy"))
groups  = np.load(os.path.join(DATA_DIR, "groups.npy"))

# CRITICAL: we need the original activity IDs (1-11), not just binary fall/non-fall.
# Most data-prep pipelines save them — check your metadata.json or prepared folder.
# Common filenames: activities.npy, activity_ids.npy, labels_multiclass.npy, y_multi.npy
candidates = ["activities.npy", "activity_ids.npy", "labels_multiclass.npy",
              "y_multi.npy", "y_activity.npy"]
activity_file = None
for c in candidates:
    if os.path.exists(os.path.join(DATA_DIR, c)):
        activity_file = c
        break

if activity_file is None:
    print("!!! No multiclass activity label file found in DATA_DIR. !!!")
    print(f"    Files present: {os.listdir(DATA_DIR)}")
    print("    You'll need to regenerate labels with activity IDs preserved.")
    print("    See note at the bottom of this cell.")
    raise FileNotFoundError("Need per-window activity IDs (1-11) to run this step.")
else:
    activities = np.load(os.path.join(DATA_DIR, activity_file))
    print(f"Loaded activities from {activity_file}")

print(f"Activity value counts:")
for a in sorted(np.unique(activities)):
    print(f"  Activity {a}: {np.sum(activities == a)} windows")

ACTIVITY_NAMES = {
    1:  "Fall: forward using hands",
    2:  "Fall: forward using knees",
    3:  "Fall: backwards",
    4:  "Fall: sideways",
    5:  "Fall: from sitting in chair",
    6:  "ADL: walking",
    7:  "ADL: standing",
    8:  "ADL: sitting",
    9:  "ADL: picking up object",
    10: "ADL: jumping",
    11: "ADL: lying",
}

# =========================
# Reproduce LOSO predictions (same settings as before)
# =========================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(RANDOM_SEED)

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
        out_dim = HIDDEN_SIZE * (2 if BIDIRECTIONAL else 1)
        self.lstm = nn.LSTM(input_size, HIDDEN_SIZE, NUM_LAYERS,
                            batch_first=True, dropout=DROPOUT,
                            bidirectional=BIDIRECTIONAL)
        self.ln = nn.LayerNorm(out_dim)
        self.att = TemporalAttention(out_dim)
        self.drop = nn.Dropout(DROPOUT)
        self.head = nn.Sequential(nn.Linear(out_dim, 64), nn.ReLU(),
                                  nn.Dropout(0.4), nn.Linear(64, 1))
    def forward(self, x):
        h, _ = self.lstm(x)
        return self.head(self.drop(self.att(self.ln(h))))

def normalise_per_channel(X_tr, X_te):
    X_tr_n, X_te_n = np.empty_like(X_tr), np.empty_like(X_te)
    for c in range(X_tr.shape[2]):
        sc = StandardScaler()
        X_tr_n[:,:,c] = sc.fit_transform(X_tr[:,:,c])
        X_te_n[:,:,c] = sc.transform(X_te[:,:,c])
    return X_tr_n, X_te_n

def train_and_predict(X_tr, y_tr, X_te, y_te):
    sss = StratifiedShuffleSplit(1, test_size=VAL_FRAC, random_state=RANDOM_SEED)
    tr, val = next(sss.split(X_tr, y_tr))
    Xa, ya, Xv, yv = X_tr[tr], y_tr[tr], X_tr[val], y_tr[val]

    n_pos = max(int(np.sum(ya == 1)), 1)
    pw = min(int(np.sum(ya == 0)) / n_pos, MAX_POS_WEIGHT)
    pw = torch.tensor([pw], dtype=torch.float32).to(DEVICE)

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
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            xb = augment_batch(xb)
            opt.zero_grad(); crit(model(xb), yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()
        vp = []
        with torch.no_grad():
            for xb,_ in vl: vp.extend(torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy().flatten())
        f1 = f1_score(yv, (np.array(vp) > 0.5).astype(int), zero_division=0)
        sch.step(f1)
        if f1 > best_f1:
            best_f1, best_state, pc = f1, copy.deepcopy(model.state_dict()), 0
        else:
            pc += 1
            if pc >= PATIENCE: break

    model.load_state_dict(best_state)
    # Tune threshold on val
    model.eval(); vp = []
    with torch.no_grad():
        for xb,_ in vl: vp.extend(torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy().flatten())
    vp = np.array(vp); best_t, best_f = 0.5, 0.0
    for t in np.arange(0.20, 0.81, 0.05):
        f = f1_score(yv, (vp > t).astype(int), zero_division=0)
        if f > best_f: best_f, best_t = f, float(t)

    tp = []
    with torch.no_grad():
        for xb,_ in el: tp.extend(torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy().flatten())
    return (np.array(tp) > best_t).astype(int), np.array(tp)

# =========================
# Run LOSO and collect per-window predictions
# =========================
with open(os.path.join(DATA_DIR, "loso_folds.json")) as f:
    folds = json.load(f)

all_rows = []
t0 = time.time()
for i, fold in enumerate(folds, 1):
    subj      = fold["test_subject"]
    train_idx = np.array(fold["train_indices"])
    test_idx  = np.array(fold["test_indices"])

    X_tr_n, X_te_n = normalise_per_channel(X[train_idx], X[test_idx])
    preds, probs = train_and_predict(X_tr_n, y[train_idx], X_te_n, y[test_idx])

    for idx, p, pr in zip(test_idx, preds, probs):
        all_rows.append({
            "subject":  int(subj),
            "window":   int(idx),
            "activity": int(activities[idx]),
            "y_true":   int(y[idx]),
            "y_pred":   int(p),
            "prob":     float(pr),
        })
    print(f"  Fold {i:>2}/{len(folds)}  Subj{subj:02d}  done  ({time.time()-t0:.1f}s)")

preds_df = pd.DataFrame(all_rows)
preds_df.to_csv(os.path.join(OUTPUT_DIR, "loso_per_window_predictions.csv"), index=False)
print(f"\nSaved per-window predictions → {len(preds_df)} rows")

# =========================
# Analyse per activity
# =========================
print("\n" + "="*80)
print("RECALL BY ACTIVITY (how often is each true activity correctly caught?)")
print("="*80)

act_summary = []
for a in sorted(preds_df["activity"].unique()):
    sub = preds_df[preds_df["activity"] == a]
    n       = len(sub)
    correct = int((sub["y_pred"] == sub["y_true"]).sum())
    if sub["y_true"].iloc[0] == 1:
        recall = (sub["y_pred"] == 1).mean()
        metric_name = "recall (catch rate)"
    else:
        recall = (sub["y_pred"] == 0).mean()
        metric_name = "specificity (correct reject)"
    act_summary.append({
        "activity": a,
        "name": ACTIVITY_NAMES.get(a, f"Activity {a}"),
        "n_windows": n,
        "accuracy": correct / n,
        "metric": metric_name,
    })

act_df = pd.DataFrame(act_summary)
print(act_df.to_string(index=False))
act_df.to_csv(os.path.join(OUTPUT_DIR, "per_activity_accuracy.csv"), index=False)

# =========================
# Per-subject × per-activity accuracy heatmap
# =========================
print("\n" + "="*80)
print("PER-SUBJECT × PER-ACTIVITY ACCURACY (this is the key plot)")
print("="*80)

subj_act = preds_df.groupby(["subject", "activity"]).apply(
    lambda g: (g["y_pred"] == g["y_true"]).mean()).unstack()

subj_act.columns = [f"A{int(c)}:\n{ACTIVITY_NAMES.get(int(c), str(c)).split(':')[-1].strip()[:15]}"
                    for c in subj_act.columns]

f1s_map = {r["test_subject"]: r["f1"] for r in
           json.load(open(os.path.join(PROJECT_ROOT,"outputs","lstm","lstm_5.6s_results.json")))["fold_results"]}
subj_act.index = [f"S{s:02d} (F1={f1s_map.get(s, 0):.2f})" for s in subj_act.index]

fig, ax = plt.subplots(figsize=(14, 9))
sns.heatmap(subj_act, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1,
            cbar_kws={"label": "accuracy"}, linewidths=0.5, ax=ax)
ax.set_title("Per-subject × per-activity accuracy (LOSO)\n"
             "Row = test subject, Col = ground-truth activity. "
             "Red = model fails on this activity for this subject.")
ax.set_xlabel(""); ax.set_ylabel("")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "subject_by_activity_accuracy.png"), dpi=120)
plt.show()

# =========================
# Which activities tend to get misclassified AS falls (false positives)?
# =========================
print("\n" + "="*80)
print("FALSE POSITIVE BREAKDOWN: when the model wrongly predicts 'fall', what was the true activity?")
print("="*80)
fp = preds_df[(preds_df["y_true"] == 0) & (preds_df["y_pred"] == 1)]
fp_by_act = fp.groupby("activity").size().sort_values(ascending=False)
for a, n in fp_by_act.items():
    total_a = (preds_df["activity"] == a).sum()
    print(f"  {ACTIVITY_NAMES.get(a, a):<35}  {n:>5} FPs  ({n/total_a:>5.1%} of all {a} windows)")

print("\nDone.")