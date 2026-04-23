"""
add_activities.py — reconstructs per-window activity IDs (1-11) for the
prepared 5.6-second window dataset.

Run AFTER prepare_upfall.py. Reads the raw CSV, re-applies the same
windowing logic, and saves activities.npy to prepared_upfall/.

The dominant activity (majority across timesteps) is assigned per window.
Binary labels and group assignments are verified to match the existing
files exactly — a mismatch would indicate drift in windowing parameters
and would invalidate per-activity analyses in Chapter 7.
"""

import os
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH    = os.path.join(PROJECT_ROOT, "data", "rawdata.csv")
OUTPUT_DIR  = os.path.join(PROJECT_ROOT, "prepared_upfall")

WINDOW_SIZE = 100
STEP_SIZE   = 50
ACTIVITY_COL = 44
SUBJECT_COL  = 43
TRIAL_COL    = 45
FALL_RATIO_THRESHOLD = 0.30

# Load existing arrays for verification
y_existing      = np.load(os.path.join(OUTPUT_DIR, "y.npy"))
groups_existing = np.load(os.path.join(OUTPUT_DIR, "groups.npy"))

# Read raw CSV
df = pd.read_csv(CSV_PATH, skiprows=2, header=None, low_memory=False)
df[[SUBJECT_COL, ACTIVITY_COL, TRIAL_COL]] = df[[SUBJECT_COL, ACTIVITY_COL, TRIAL_COL]].apply(pd.to_numeric, errors="coerce")
df = df.dropna(subset=[SUBJECT_COL, ACTIVITY_COL, TRIAL_COL]).reset_index(drop=True)

activities = []
y_rebuilt  = []
groups_rebuilt = []

for (subj, trial), segment in df.groupby([SUBJECT_COL, TRIAL_COL]):
    segment = segment.reset_index(drop=True)
    if len(segment) < WINDOW_SIZE:
        continue
    for start in range(0, len(segment) - WINDOW_SIZE + 1, STEP_SIZE):
        end = start + WINDOW_SIZE
        activity_values = segment.iloc[start:end][ACTIVITY_COL].values.astype(int)

        # Dominant activity = most frequent activity ID in the window
        unique, counts = np.unique(activity_values, return_counts=True)
        dominant_activity = int(unique[counts.argmax()])
        activities.append(dominant_activity)

        # Regenerate binary label for verification
        fall_ratio = np.mean((activity_values >= 1) & (activity_values <= 5))
        y_rebuilt.append(1 if fall_ratio >= FALL_RATIO_THRESHOLD else 0)
        groups_rebuilt.append(int(subj))

activities = np.array(activities, dtype=np.int64)
y_rebuilt  = np.array(y_rebuilt, dtype=np.int64)
groups_rebuilt = np.array(groups_rebuilt, dtype=np.int64)

# Sanity checks: binary labels and groups must match the existing preparation exactly
assert len(activities) == len(y_existing), f"Window count mismatch: {len(activities)} vs {len(y_existing)}"
assert np.array_equal(y_rebuilt, y_existing), "Binary labels do not match!"
assert np.array_equal(groups_rebuilt, groups_existing), "Subject IDs do not match!"

np.save(os.path.join(OUTPUT_DIR, "activities.npy"), activities)
print(f"Saved {len(activities)} activity labels to {OUTPUT_DIR}/activities.npy")
print(f"Activity distribution: {np.bincount(activities)}")