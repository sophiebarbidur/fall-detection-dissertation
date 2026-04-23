import os
import json
import numpy as np
import pandas as pd

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# =========================================================
# PATHS
# =========================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "rawdata.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "prepared_upfall_2s")   # new folder, keeps old data safe

# =========================================================
# CONFIG
# =========================================================
# FIX 1: Sampling rate.
# Your original script said 50 Hz but the UP-Fall dataset is ~18 Hz.
# This was not causing a crash but it was wrong in the metadata and
# was the reason your windows were 5.6 s instead of 2 s:
#   100 steps / 18 Hz = 5.6 s  (what you actually had)
#   100 steps / 50 Hz = 2.0 s  (what your comment claimed)
# The raw data does not change — only the interpretation and the
# window/step sizes derived from it change.
SAMPLING_RATE_HZ = 18

# FIX 2: Window size.
# 2 seconds × 18 Hz = 36 timesteps.
# The paper explicitly found 2-second windows optimal.
# At 5.6 s the fall event (which lasts ~1-2 s) is a minority of the
# window, so the model sees mostly normal movement even in fall windows.
# At 2 s the fall event dominates the window — much cleaner signal.
WINDOW_SIZE = 36    # 2 seconds at 18 Hz

# FIX 3: Step size (overlap).
# 50% overlap = step of WINDOW_SIZE // 2 = 18 timesteps.
# Your original 50-step stride on 100-step windows was also 50% —
# so the overlap fraction is unchanged, just scaled to the new window.
# More overlap = more windows = more training data, which helps given
# how few fall events there are. Don't go below 50% or you risk
# the fall event falling entirely in the gap between windows.
STEP_SIZE = 18      # 50% overlap

# =========================================================
# COLUMN MAP  (unchanged — your original was correct)
# =========================================================
SENSORS = {
    "ankle": {
        "acc":  [1, 2, 3],
        "gyro": [4, 5, 6]
    },
    "pocket": {
        "acc":  [8, 9, 10],
        "gyro": [11, 12, 13]
    },
    "belt": {
        "acc":  [15, 16, 17],
        "gyro": [18, 19, 20]
    },
    "neck": {
        "acc":  [22, 23, 24],
        "gyro": [25, 26, 27]
    },
    "wrist": {
        "acc":  [29, 30, 31],
        "gyro": [32, 33, 34]
    }
}

SUBJECT_COL  = 43
ACTIVITY_COL = 44
TRIAL_COL    = 45
TAG_COL      = 46

# FIX 4: Fall ratio threshold.
# Your original threshold was 0.30, meaning a window only needed 30%
# of its timesteps to be a fall to get labelled as fall.
# At 5.6-second windows that was already generous.
# At 2-second windows it is even more important to be precise:
# with 36 timesteps you want the fall to genuinely dominate the window,
# not just clip the edge of it. 0.50 means at least half the window
# must be fall activity — this gives you cleaner labels and reduces
# ambiguous boundary windows that confuse the model.
# If you find you lose too many fall windows, you can lower this back
# to 0.40, but start at 0.50.
FALL_LABEL_MIN        = 1
FALL_LABEL_MAX        = 5
FALL_RATIO_THRESHOLD  = 0.50

# =========================================================
# HELPERS
# =========================================================
def get_feature_columns_and_names():
    feature_cols  = []
    feature_names = []
    for sensor_name, parts in SENSORS.items():
        for sensor_type in ["acc", "gyro"]:
            for axis_name, col_idx in zip(["x", "y", "z"], parts[sensor_type]):
                feature_cols.append(col_idx)
                feature_names.append(f"{sensor_name}_{sensor_type}_{axis_name}")
    return feature_cols, feature_names

def assign_window_label(activity_values):
    """
    1 = fall  if ≥ FALL_RATIO_THRESHOLD of timesteps are fall activity
    0 = no fall otherwise
    """
    activity_values = np.asarray(activity_values)
    fall_ratio = np.mean(
        (activity_values >= FALL_LABEL_MIN) & (activity_values <= FALL_LABEL_MAX)
    )
    return 1 if fall_ratio >= FALL_RATIO_THRESHOLD else 0

# =========================================================
# MAIN
# =========================================================
def main():
    print("Loading UP-Fall data...")
    print(f"CSV path: {CSV_PATH}")

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH, skiprows=2, header=None, low_memory=False)
    print(f"Raw dataframe shape: {df.shape}")

    feature_cols, feature_names = get_feature_columns_and_names()
    required_cols = feature_cols + [SUBJECT_COL, ACTIVITY_COL, TRIAL_COL, TAG_COL]

    print("Converting required columns to numeric...")
    df[required_cols] = df[required_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=required_cols).reset_index(drop=True)
    print(f"Rows after cleaning: {len(df)}")

    print(f"\nWindow size : {WINDOW_SIZE} timesteps = {WINDOW_SIZE/SAMPLING_RATE_HZ:.1f} s")
    print(f"Step size   : {STEP_SIZE} timesteps  = {STEP_SIZE/SAMPLING_RATE_HZ:.1f} s")
    print(f"Overlap     : {(1 - STEP_SIZE/WINDOW_SIZE)*100:.0f}%")
    print(f"Fall threshold: {FALL_RATIO_THRESHOLD*100:.0f}% of window\n")

    X_windows = []
    y         = []
    groups    = []

    print("Creating windows (per subject+trial, no cross-boundary mixing)...")

    for (subj, trial), segment in df.groupby([SUBJECT_COL, TRIAL_COL]):
        segment = segment.reset_index(drop=True)

        if len(segment) < WINDOW_SIZE:
            continue

        for start in range(0, len(segment) - WINDOW_SIZE + 1, STEP_SIZE):
            end    = start + WINDOW_SIZE
            window = segment.iloc[start:end]

            label    = assign_window_label(window[ACTIVITY_COL].values)
            X_window = window[feature_cols].values.astype(np.float32)

            X_windows.append(X_window)
            y.append(label)
            groups.append(int(subj))

    X_windows = np.array(X_windows, dtype=np.float32)
    y         = np.array(y,         dtype=np.int64)
    groups    = np.array(groups)

    if len(X_windows) == 0:
        raise RuntimeError("No windows created — check data path and column indices.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR, "X_windows.npy"), X_windows)
    np.save(os.path.join(OUTPUT_DIR, "y.npy"),         y)
    np.save(os.path.join(OUTPUT_DIR, "groups.npy"),    groups)

    unique_subjects = sorted(np.unique(groups).tolist())
    folds = []
    for test_subj in unique_subjects:
        test_idx  = np.where(groups == test_subj)[0].tolist()
        train_idx = np.where(groups != test_subj)[0].tolist()
        folds.append({
            "test_subject":  int(test_subj),
            "train_indices": train_idx,
            "test_indices":  test_idx
        })

    with open(os.path.join(OUTPUT_DIR, "loso_folds.json"), "w") as f:
        json.dump(folds, f)
    print(f"Saved {len(folds)} LOSO fold definitions")

    metadata = {
        "csv_path":               CSV_PATH,
        "sampling_rate_hz":       SAMPLING_RATE_HZ,
        "window_size":            WINDOW_SIZE,
        "step_size":              STEP_SIZE,
        "window_duration_seconds": WINDOW_SIZE / SAMPLING_RATE_HZ,
        "overlap_fraction":       1 - (STEP_SIZE / WINDOW_SIZE),
        "fall_ratio_threshold":   FALL_RATIO_THRESHOLD,
        "num_windows":            int(len(X_windows)),
        "window_shape":           list(X_windows.shape[1:]),
        "num_fall_windows":       int(np.sum(y == 1)),
        "num_no_fall_windows":    int(np.sum(y == 0)),
        "unique_subjects":        unique_subjects,
        "num_subjects":           len(unique_subjects),
        "feature_names":          feature_names,
        "sensors":                SENSORS,
    }

    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # ── sanity check ──────────────────────────────────────
    imbalance = np.sum(y == 0) / max(np.sum(y == 1), 1)
    print("\n--- DONE ---")
    print(f"Output dir   : {OUTPUT_DIR}")
    print(f"X_windows    : {X_windows.shape}  (N, timesteps, channels)")
    print(f"Fall windows : {np.sum(y == 1)}")
    print(f"Non-fall     : {np.sum(y == 0)}")
    print(f"Imbalance    : {imbalance:.1f}:1  (non-fall : fall)")
    print(f"Subjects     : {unique_subjects}")
    print(f"\nNow point DATA_DIR in your model scripts to:")
    print(f"  {OUTPUT_DIR}")

if __name__ == "__main__":
    main()