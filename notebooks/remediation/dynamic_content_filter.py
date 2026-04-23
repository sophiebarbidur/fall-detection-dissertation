# =========================
# Step 4: Dynamic content filter — post-hoc rule
# =========================
import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(PROJECT_ROOT, "prepared_upfall")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "outputs", "diagnostics")

X          = np.load(os.path.join(DATA_DIR, "X_windows.npy"))
activities = np.load(os.path.join(DATA_DIR, "activities.npy"))
preds_df   = pd.read_csv(os.path.join(OUTPUT_DIR, "loso_per_window_predictions.csv"))

# =========================
# Motion energy per window
# =========================
def motion_energy(w):
    return float(np.mean(np.std(w, axis=0)))

preds_df["motion_energy"] = preds_df["window"].apply(lambda i: motion_energy(X[i]))

# =========================
# Baseline (no filter)
# =========================
y_true = preds_df["y_true"].values
y_base = preds_df["y_pred"].values
base_f1   = f1_score(y_true, y_base, zero_division=0)
base_acc  = accuracy_score(y_true, y_base)
base_prec = precision_score(y_true, y_base, zero_division=0)
base_rec  = recall_score(y_true, y_base, zero_division=0)

print("BASELINE (no filter):")
print(f"  acc={base_acc:.2%}  f1={base_f1:.3f}  prec={base_prec:.3f}  rec={base_rec:.3f}")
print(f"  confusion matrix:\n{confusion_matrix(y_true, y_base)}")

# =========================
# Sweep: for a range of motion-energy thresholds, override predictions
# =========================
energies = preds_df["motion_energy"].values
thresholds = np.percentile(energies[y_base == 1], np.arange(0, 51, 5))

sweep_rows = []
for thr in thresholds:
    # Rule: if predicted fall AND motion_energy < thr, override to non-fall
    y_filt = y_base.copy()
    override_mask = (y_filt == 1) & (energies < thr)
    y_filt[override_mask] = 0

    # How many true falls did we accidentally kill?
    falls_killed = int(((preds_df["y_true"] == 1) & override_mask).sum())
    fps_killed   = int(((preds_df["y_true"] == 0) & override_mask).sum())

    sweep_rows.append({
        "threshold":   float(thr),
        "n_overrides": int(override_mask.sum()),
        "falls_killed": falls_killed,
        "fps_killed":   fps_killed,
        "accuracy":  float(accuracy_score(y_true, y_filt)),
        "precision": float(precision_score(y_true, y_filt, zero_division=0)),
        "recall":    float(recall_score(y_true, y_filt, zero_division=0)),
        "f1":        float(f1_score(y_true, y_filt, zero_division=0)),
    })

sweep_df = pd.DataFrame(sweep_rows)
print("\nFilter threshold sweep:")
print(sweep_df.to_string(index=False))

# Best threshold by F1
best = sweep_df.loc[sweep_df["f1"].idxmax()]
print(f"\nBest threshold: {best['threshold']:.3f}")
print(f"  F1:      {base_f1:.3f} → {best['f1']:.3f}  (Δ {best['f1'] - base_f1:+.3f})")
print(f"  Accuracy:{base_acc:.2%} → {best['accuracy']:.2%}  (Δ {best['accuracy'] - base_acc:+.2%})")
print(f"  Precision: {base_prec:.3f} → {best['precision']:.3f}")
print(f"  Recall:    {base_rec:.3f} → {best['recall']:.3f}")
print(f"  Killed {int(best['fps_killed'])} FPs at cost of {int(best['falls_killed'])} missed falls")

# =========================
# Plot: precision/recall tradeoff
# =========================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(sweep_df["threshold"], sweep_df["precision"], "o-", label="precision", color="blue")
axes[0].plot(sweep_df["threshold"], sweep_df["recall"], "o-", label="recall", color="red")
axes[0].plot(sweep_df["threshold"], sweep_df["f1"], "o-", label="f1", color="green")
axes[0].axhline(base_f1, color="green", linestyle="--", alpha=0.5, label="baseline F1")
axes[0].axvline(best["threshold"], color="black", linestyle=":", alpha=0.5)
axes[0].set_xlabel("Motion energy threshold")
axes[0].set_ylabel("Metric")
axes[0].set_title("Filter threshold sweep")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(sweep_df["fps_killed"], sweep_df["falls_killed"], "o-")
for _, r in sweep_df.iterrows():
    axes[1].annotate(f"{r['threshold']:.2f}", (r["fps_killed"], r["falls_killed"]),
                     fontsize=7, xytext=(3, 3), textcoords="offset points")
axes[1].set_xlabel("False positives removed")
axes[1].set_ylabel("True falls lost (cost)")
axes[1].set_title("Tradeoff: benefit vs cost of filter")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "dynamic_filter_sweep.png"), dpi=120)
plt.show()

# =========================
# At the best threshold, per-activity breakdown
# =========================
thr = best["threshold"]
preds_df["y_filtered"] = preds_df["y_pred"].copy()
preds_df.loc[(preds_df["y_pred"] == 1) & (preds_df["motion_energy"] < thr), "y_filtered"] = 0

print(f"\nPer-activity impact at threshold {thr:.3f}:")
ACTIVITY_NAMES = {
    1: "Fall: forward/hands", 2: "Fall: forward/knees", 3: "Fall: backwards",
    4: "Fall: sideways",      5: "Fall: from sitting",
    6: "ADL: walking",        7: "ADL: standing",       8: "ADL: sitting",
    9: "ADL: picking up",    10: "ADL: jumping",       11:"ADL: lying",
}
for a in sorted(preds_df["activity"].unique()):
    sub = preds_df[preds_df["activity"] == a]
    before_wrong = int((sub["y_pred"]     != sub["y_true"]).sum())
    after_wrong  = int((sub["y_filtered"] != sub["y_true"]).sum())
    fixed = before_wrong - after_wrong
    print(f"  {ACTIVITY_NAMES.get(a, f'A{a}'):<22}  wrong before={before_wrong:>4}  "
          f"wrong after={after_wrong:>4}  fixed={fixed:>+4}")

# Save
sweep_df.to_csv(os.path.join(OUTPUT_DIR, "dynamic_filter_sweep.csv"), index=False)
preds_df.to_csv(os.path.join(OUTPUT_DIR, "loso_predictions_with_filter.csv"), index=False)
print(f"\nSaved → dynamic_filter_sweep.csv, loso_predictions_with_filter.csv")