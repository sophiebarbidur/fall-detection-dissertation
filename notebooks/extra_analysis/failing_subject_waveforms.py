# =========================
# Step 2: Why do S07 and S16 fail on specific fall types?
# =========================
import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(PROJECT_ROOT, "prepared_upfall")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "outputs", "diagnostics")

X          = np.load(os.path.join(DATA_DIR, "X_windows.npy"))
y          = np.load(os.path.join(DATA_DIR, "y.npy"))
groups     = np.load(os.path.join(DATA_DIR, "groups.npy"))
activities = np.load(os.path.join(DATA_DIR, "activities.npy"))

with open(os.path.join(DATA_DIR, "metadata.json")) as f:
    feature_names = json.load(f)["feature_names"]

ACTIVITY_NAMES = {
    1: "Forward/hands", 2: "Forward/knees", 3: "Backwards",
    4: "Sideways",      5: "From sitting",
}

# Compare each failing subject's falls to the pool of other subjects' falls
def signal_profile(windows):
    """Return per-channel mean peak acc magnitude and mean std across time."""
    peaks = np.max(np.abs(windows), axis=1)           # (n, C)
    stds  = np.std(windows, axis=1)                   # (n, C)
    return peaks.mean(axis=0), stds.mean(axis=0)

cases = [
    ("S07", 7,  [2, 4, 5], "Forward/knees, sideways, from-sitting fail"),
    ("S16", 16, [1, 2],    "Forward/hands, forward/knees fail"),
    ("S13", 13, [6, 7, 8, 9, 10], "All ADLs classified as falls"),
    ("S10", 10, [8, 9, 10, 11],   "Multiple ADLs misclassified"),
]

for tag, subj, target_activities, desc in cases:
    print(f"\n{'='*70}")
    print(f"{tag} (Subject {subj}) — {desc}")
    print(f"{'='*70}")

    for act in target_activities:
        this_mask  = (groups == subj) & (activities == act)
        other_mask = (groups != subj) & (activities == act)

        n_this  = int(this_mask.sum())
        n_other = int(other_mask.sum())
        if n_this == 0 or n_other == 0:
            print(f"  Activity {act}: insufficient data (this={n_this}, other={n_other})")
            continue

        this_peaks, this_stds   = signal_profile(X[this_mask])
        other_peaks, other_stds = signal_profile(X[other_mask])

        # Summary: how does this subject's signal differ from the pool?
        peak_ratio = this_peaks.mean() / (other_peaks.mean() + 1e-8)
        std_ratio  = this_stds.mean()  / (other_stds.mean()  + 1e-8)

        act_name = ACTIVITY_NAMES.get(act, f"Activity{act}")
        print(f"  [{act_name:>15}] n_this={n_this:>3}  n_other={n_other:>4}  "
              f"peak_ratio={peak_ratio:.2f}  std_ratio={std_ratio:.2f}")

        # Channels where this subject differs most from the pool
        peak_diff = (this_peaks - other_peaks) / (other_peaks + 1e-8)
        top_diff  = np.argsort(np.abs(peak_diff))[-3:][::-1]
        for ch in top_diff:
            print(f"      {feature_names[ch]:<20}  this={this_peaks[ch]:>8.1f}  "
                  f"others={other_peaks[ch]:>8.1f}  diff={peak_diff[ch]:+.1%}")

# =========================
# Visual: one failing subject's falls vs the pool
# =========================
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
for ax, (subj, act) in zip(axes.flatten(),
                           [(7, 2), (7, 4), (16, 1), (13, 6)]):
    this_mask  = (groups == subj) & (activities == act)
    other_mask = (groups != subj) & (activities == act)
    if this_mask.sum() == 0:
        continue

    # Plot magnitude of waist accelerometer (belt_acc_*) as overall body motion proxy
    # belt_acc_x/y/z are channels 12,13,14 in your feature list
    def accel_mag(windows):
        return np.sqrt(windows[:, :, 12]**2 + windows[:, :, 13]**2 + windows[:, :, 14]**2)

    this_mag  = accel_mag(X[this_mask])
    other_mag = accel_mag(X[other_mask])

    # Plot mean ± band for others, individual lines for this subject
    t = np.arange(100)
    ax.fill_between(t, other_mag.mean(0) - other_mag.std(0),
                       other_mag.mean(0) + other_mag.std(0),
                    alpha=0.25, color="grey", label=f"Other subjects (n={len(other_mag)})")
    ax.plot(t, other_mag.mean(0), color="grey", linewidth=2)
    for i in range(min(5, len(this_mag))):
        ax.plot(t, this_mag[i], color="red", alpha=0.5, linewidth=1,
                label=f"S{subj} (n={len(this_mag)})" if i == 0 else None)

    act_name = ACTIVITY_NAMES.get(act, str(act))
    ax.set_title(f"S{subj:02d} — {act_name} (waist accel magnitude)")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("|acc|")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

plt.suptitle("Failing subjects' fall waveforms (red) vs the rest of the training pool (grey band)",
             fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "failing_subject_waveforms.png"), dpi=120)
plt.show()