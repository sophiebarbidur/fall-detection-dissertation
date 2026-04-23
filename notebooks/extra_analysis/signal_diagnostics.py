# =========================
# Step 1: Per-Subject Signal Diagnostics
# =========================
import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import jensenshannon

# =========================
# Paths (same as main notebook)
# =========================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(PROJECT_ROOT, "prepared_upfall")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "outputs", "diagnostics")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# Load data
# =========================
print("Loading dataset...")
X      = np.load(os.path.join(DATA_DIR, "X_windows.npy"))       # (N, T, C)
y      = np.load(os.path.join(DATA_DIR, "y.npy"))               # (N,)
groups = np.load(os.path.join(DATA_DIR, "groups.npy"))          # (N,) subject IDs

# Load LOSO results so we can cross-reference
loso_path = os.path.join(PROJECT_ROOT, "outputs", "lstm", "lstm_5.6s_results.json")
with open(loso_path) as f:
    loso_results = json.load(f)
subj_to_acc = {r["test_subject"]: r["accuracy"] for r in loso_results["fold_results"]}
subj_to_f1  = {r["test_subject"]: r["f1"]       for r in loso_results["fold_results"]}

N, T, C = X.shape
print(f"X shape: {X.shape}")
print(f"Unique subjects: {sorted(np.unique(groups).tolist())}")
print(f"Total falls: {int(np.sum(y==1))}, non-falls: {int(np.sum(y==0))}\n")

# =========================
# 1) Per-subject basic stats
# =========================
print("="*80)
print("PER-SUBJECT SUMMARY")
print("="*80)

rows = []
for subj in sorted(np.unique(groups)):
    mask   = groups == subj
    X_sub  = X[mask]
    y_sub  = y[mask]
    n_win  = len(X_sub)
    n_fall = int(np.sum(y_sub == 1))
    n_nf   = int(np.sum(y_sub == 0))
    # Flatten (n_windows * timesteps, channels) to get per-channel distribution
    flat   = X_sub.reshape(-1, C)
    rows.append({
        "subject":   int(subj),
        "n_windows": n_win,
        "n_falls":   n_fall,
        "n_nonfalls": n_nf,
        "fall_rate": n_fall / max(n_win, 1),
        "mean_abs":  float(np.mean(np.abs(flat))),
        "std_all":   float(np.std(flat)),
        "max_abs":   float(np.max(np.abs(flat))),
        "loso_acc":  subj_to_acc.get(int(subj), np.nan),
        "loso_f1":   subj_to_f1.get(int(subj), np.nan),
    })

df = pd.DataFrame(rows).sort_values("loso_f1")
print(df.to_string(index=False))
df.to_csv(os.path.join(OUTPUT_DIR, "per_subject_summary.csv"), index=False)
print(f"\nSaved → {os.path.join(OUTPUT_DIR, 'per_subject_summary.csv')}")

# =========================
# 2) Per-subject, per-channel mean & std
# =========================
print("\n" + "="*80)
print("PER-SUBJECT, PER-CHANNEL STATS (mean & std)")
print("="*80)

subjects = sorted(np.unique(groups).tolist())
n_subj   = len(subjects)

mean_mat = np.zeros((n_subj, C))
std_mat  = np.zeros((n_subj, C))

for i, subj in enumerate(subjects):
    mask = groups == subj
    flat = X[mask].reshape(-1, C)
    mean_mat[i] = flat.mean(axis=0)
    std_mat[i]  = flat.std(axis=0)

# Identify outlier subjects: how far is each subject's per-channel mean
# from the global (across-subject) mean, in units of across-subject std?
global_mean = mean_mat.mean(axis=0)
global_std  = mean_mat.std(axis=0) + 1e-8
z_means     = (mean_mat - global_mean) / global_std       # (n_subj, C)
outlier_score = np.mean(np.abs(z_means), axis=1)          # per-subject scalar

outlier_df = pd.DataFrame({
    "subject": subjects,
    "mean_z_deviation": outlier_score,
    "loso_f1": [subj_to_f1.get(s, np.nan) for s in subjects],
    "loso_acc": [subj_to_acc.get(s, np.nan) for s in subjects],
}).sort_values("mean_z_deviation", ascending=False)

print("\nSubjects ranked by signal outlier-ness (higher = more unusual):")
print(outlier_df.to_string(index=False))
outlier_df.to_csv(os.path.join(OUTPUT_DIR, "outlier_ranking.csv"), index=False)

# Correlation between outlier-ness and LOSO F1
corr = outlier_df[["mean_z_deviation", "loso_f1"]].corr().iloc[0, 1]
print(f"\nCorrelation(outlier_score, LOSO F1) = {corr:.3f}")
print("(Negative correlation = unusual subjects perform worse, which is what we'd expect)")

# =========================
# 3) Plots
# =========================
sns.set_style("whitegrid")

# Plot A: LOSO F1 vs outlier score
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(outlier_df["mean_z_deviation"], outlier_df["loso_f1"], s=80, alpha=0.7)
for _, row in outlier_df.iterrows():
    ax.annotate(f"S{int(row['subject']):02d}",
                (row["mean_z_deviation"], row["loso_f1"]),
                xytext=(5, 5), textcoords="offset points", fontsize=9)
ax.set_xlabel("Mean |z-deviation| from cross-subject average")
ax.set_ylabel("LOSO F1 score")
ax.set_title(f"Signal outlier-ness vs LOSO F1  (corr = {corr:.2f})")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "outlier_vs_f1.png"), dpi=120)
plt.show()

# Plot B: Heatmap of per-subject, per-channel means (z-scored)
fig, ax = plt.subplots(figsize=(14, 8))
sns.heatmap(z_means,
            yticklabels=[f"S{s:02d} (F1={subj_to_f1.get(s, np.nan):.2f})" for s in subjects],
            xticklabels=[f"Ch{c}" for c in range(C)],
            cmap="RdBu_r", center=0, vmin=-3, vmax=3, ax=ax,
            cbar_kws={"label": "z-score vs cross-subject mean"})
ax.set_title("Per-subject, per-channel signal mean (z-scored across subjects)\n"
             "Red/blue = this subject's channel is unusually high/low vs others")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "zscore_heatmap.png"), dpi=120)
plt.show()

# Plot C: Distribution comparison for worst vs best subjects on a few key channels
worst_subjects = outlier_df.nsmallest(3, "loso_f1")["subject"].tolist()
best_subjects  = outlier_df.nlargest(3, "loso_f1")["subject"].tolist()
print(f"\nWorst LOSO subjects: {worst_subjects}")
print(f"Best LOSO subjects:  {best_subjects}")

# Pick 6 channels to visualize (spread across the 30)
channels_to_plot = [0, 5, 10, 15, 20, 25]
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, ch in zip(axes.flatten(), channels_to_plot):
    for subj in worst_subjects:
        vals = X[groups == subj][:, :, ch].flatten()
        ax.hist(vals, bins=60, alpha=0.4, density=True,
                label=f"S{subj:02d} (worst, F1={subj_to_f1[subj]:.2f})",
                color="red")
    for subj in best_subjects:
        vals = X[groups == subj][:, :, ch].flatten()
        ax.hist(vals, bins=60, alpha=0.4, density=True,
                label=f"S{subj:02d} (best, F1={subj_to_f1[subj]:.2f})",
                color="blue")
    ax.set_title(f"Channel {ch}")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend(fontsize=7)
plt.suptitle("Signal distributions: worst LOSO subjects (red) vs best (blue)", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "worst_vs_best_distributions.png"), dpi=120)
plt.show()

# =========================
# 4) Jensen-Shannon divergence: how different is each subject from the "rest"?
# =========================
print("\n" + "="*80)
print("JENSEN-SHANNON DIVERGENCE (each subject vs all others combined)")
print("="*80)
print("Higher value = this subject's signal distribution is more different from the pool.\n")

def js_divergence_1d(a, b, bins=50):
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    edges = np.linspace(lo, hi, bins + 1)
    p, _ = np.histogram(a, bins=edges, density=True)
    q, _ = np.histogram(b, bins=edges, density=True)
    p = p + 1e-10
    q = q + 1e-10
    p = p / p.sum()
    q = q / q.sum()
    return jensenshannon(p, q)

js_rows = []
for subj in subjects:
    in_mask  = groups == subj
    out_mask = ~in_mask
    divs = []
    for ch in range(C):
        a = X[in_mask][:, :, ch].flatten()
        b = X[out_mask][:, :, ch].flatten()
        # Subsample to keep this fast
        if len(a) > 20000: a = np.random.choice(a, 20000, replace=False)
        if len(b) > 20000: b = np.random.choice(b, 20000, replace=False)
        divs.append(js_divergence_1d(a, b))
    js_rows.append({
        "subject": int(subj),
        "mean_JS_div": float(np.mean(divs)),
        "max_JS_div": float(np.max(divs)),
        "loso_f1": subj_to_f1.get(int(subj), np.nan),
    })

js_df = pd.DataFrame(js_rows).sort_values("mean_JS_div", ascending=False)
print(js_df.to_string(index=False))
js_df.to_csv(os.path.join(OUTPUT_DIR, "js_divergence.csv"), index=False)

corr_js = js_df[["mean_JS_div", "loso_f1"]].corr().iloc[0, 1]
print(f"\nCorrelation(mean JS divergence, LOSO F1) = {corr_js:.3f}")

# =========================
# 5) Summary
# =========================
print("\n" + "="*80)
print("SUMMARY — what to look for")
print("="*80)
print("""
1. If mean_z_deviation and mean_JS_div are NEGATIVELY correlated with LOSO F1,
   then subjects whose signals look distributionally different from the training
   pool are the ones your model fails on. This confirms the generalization
   hypothesis: the model isn't learning subject-invariant fall patterns.

2. Look at the z-score heatmap. Do the failing subjects (10, 13, 14, 16, 17)
   have entire rows that are unusually red or blue? Is it one particular sensor
   channel that's off (e.g. maybe they wore the wrist IMU differently)?

3. Look at the distribution comparison plot. Are the worst subjects' histograms
   shifted, wider, or multi-modal compared to the best?

4. If there's NO clear signal difference but LOSO F1 is still bad, the problem
   is more subtle — likely movement style / timing rather than raw amplitude.
   In that case, per-subject normalization won't help as much, and you'd want
   to look at derived features (velocity, acceleration magnitude).
""")