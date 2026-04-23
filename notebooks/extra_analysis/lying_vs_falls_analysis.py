# =========================
# Step 1: Visual comparison — falls vs correctly-rejected lying vs misclassified lying
# =========================
import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(PROJECT_ROOT, "prepared_upfall")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "outputs", "diagnostics")

X          = np.load(os.path.join(DATA_DIR, "X_windows.npy"))
activities = np.load(os.path.join(DATA_DIR, "activities.npy"))
preds_df   = pd.read_csv(os.path.join(OUTPUT_DIR, "loso_per_window_predictions.csv"))

with open(os.path.join(DATA_DIR, "metadata.json")) as f:
    feature_names = json.load(f)["feature_names"]

# =========================
# Pull three groups of windows
# =========================
# a) Correctly classified falls (any fall type)
# b) Correctly rejected lying
# c) Misclassified lying (predicted fall, was lying)
correct_falls = preds_df[(preds_df["y_true"] == 1) & (preds_df["y_pred"] == 1)
                         & (preds_df["activity"].between(1, 5))]
correct_lying = preds_df[(preds_df["activity"] == 11) & (preds_df["y_pred"] == 0)]
missed_lying  = preds_df[(preds_df["activity"] == 11) & (preds_df["y_pred"] == 1)]

print(f"Correctly classified falls : {len(correct_falls)}")
print(f"Correctly rejected lying   : {len(correct_lying)}")
print(f"Misclassified lying (FP)   : {len(missed_lying)}")

# Sample a few windows from each
rng = np.random.RandomState(42)
n_examples = 4
idx_falls  = correct_falls.sample(n_examples, random_state=42)["window"].values
idx_clying = correct_lying.sample(n_examples, random_state=42)["window"].values
idx_mlying = missed_lying.sample(n_examples, random_state=42)["window"].values

# =========================
# Compute per-window motion energy (std across time, summed across channels)
# =========================
def motion_energy(w):
    # w: (T, C). Std over time for each channel, then mean
    return float(np.mean(np.std(w, axis=0)))

me_falls  = [motion_energy(X[i]) for i in correct_falls["window"]]
me_clying = [motion_energy(X[i]) for i in correct_lying["window"]]
me_mlying = [motion_energy(X[i]) for i in missed_lying["window"]]

print(f"\nMotion energy (mean ± std):")
print(f"  Correct falls         : {np.mean(me_falls):.2f} ± {np.std(me_falls):.2f}")
print(f"  Correctly rejected lying: {np.mean(me_clying):.2f} ± {np.std(me_clying):.2f}")
print(f"  Misclassified lying   : {np.mean(me_mlying):.2f} ± {np.std(me_mlying):.2f}")

# =========================
# Plot A: Motion energy distributions (the key diagnostic)
# =========================
fig, ax = plt.subplots(figsize=(10, 5))
bins = np.linspace(0, max(max(me_falls), max(me_clying), max(me_mlying)), 60)
ax.hist(me_clying, bins=bins, alpha=0.5, label=f"Correctly rejected lying (n={len(me_clying)})",
        color="green", density=True)
ax.hist(me_mlying, bins=bins, alpha=0.5, label=f"MISCLASSIFIED lying as fall (n={len(me_mlying)})",
        color="red", density=True)
ax.hist(me_falls,  bins=bins, alpha=0.4, label=f"Correct falls (n={len(me_falls)})",
        color="blue", density=True)
ax.set_xlabel("Motion energy (mean std across channels)")
ax.set_ylabel("Density")
ax.set_title("Motion energy distribution — do misclassified lying windows look like falls?")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "motion_energy_distributions.png"), dpi=120)
plt.show()

# =========================
# Plot B: Example waveforms (4 channels, 3 window types, 4 examples each)
# =========================
# Pick 4 informative channels — one accelerometer from each sensor group roughly
channels_to_plot = [0, 6, 12, 24]  # ankle_acc_x, pocket_acc_x, belt_acc_x, wrist_acc_x

fig, axes = plt.subplots(3, n_examples, figsize=(4 * n_examples, 9), sharey="row")
row_titles = ["Correct FALL (true fall)", "Correctly rejected LYING",
              "MISCLASSIFIED LYING (predicted fall)"]
idx_groups = [idx_falls, idx_clying, idx_mlying]
row_colors = ["blue", "green", "red"]

for row, (title, idxs, color) in enumerate(zip(row_titles, idx_groups, row_colors)):
    for col, idx in enumerate(idxs):
        w = X[idx]  # (100, 30)
        ax = axes[row, col]
        for ch in channels_to_plot:
            ax.plot(w[:, ch], alpha=0.7, linewidth=1.2,
                    label=feature_names[ch] if (row == 0 and col == 0) else None)
        ax.set_title(f"{title}\nw{idx}, S{int(preds_df.loc[preds_df['window']==idx, 'subject'].iloc[0])}",
                     fontsize=9, color=color)
        ax.grid(alpha=0.3)
        if col == 0:
            ax.set_ylabel("Signal value")
        if row == 2:
            ax.set_xlabel("Timestep")

axes[0, 0].legend(fontsize=7, loc="upper right")
plt.suptitle("Example windows — do misclassified lying windows (bottom row) look like falls (top row)?",
             fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "example_waveforms.png"), dpi=120)
plt.show()

print("\nInterpretation:")
print("- If motion energy of misclassified lying overlaps with correct falls → story confirmed")
print("- If example misclassified lying waveforms look flat (like correctly rejected lying),")
print("  then the model is being overly aggressive regardless of signal content")