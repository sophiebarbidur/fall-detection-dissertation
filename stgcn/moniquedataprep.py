"""
gnn_data_prep.py  —  UP-Fall Dataset Loader
Aligned with Yan et al. (2023):
  - 2-second windows at 18 Hz = 36 time steps
  - 50% overlap (step = 18)
  - Multiclass labels 0-10 (11 activities)
  - Subject IDs returned for proper subject-level splitting
  - Transition frames (label=20) skipped
  - Node order matches skeleton: ankle(S1), pocket(S2), waist(S3), neck(S4), wrist(S5)
"""

import pandas as pd
import numpy as np

# -------------------------------------------------
# WINDOWING CONFIG
# Must match the ST-GCN training script:
#   WINDOW_SEC=2.0, SAMPLE_RATE=18 -> 36 time steps
#   OVERLAP=0.5 -> step = 18
# -------------------------------------------------
SAMPLE_RATE = 18
WINDOW_SIZE = int(2.0 * SAMPLE_RATE)   # 36 time steps
STEP        = int(WINDOW_SIZE * 0.5)   # 18 (50% overlap)

# -------------------------------------------------
# SENSOR COLUMN INDICES  (1-based Signal IDs from the UP-Fall paper)
# Each node: [Ax, Ay, Az, Gx, Gy, Gz]  — luminosity (col 7/14/21/28/35) dropped
# -------------------------------------------------
SENSOR_MAP = {
    "ankle":  [1,  2,  3,  4,  5,  6],
    "pocket": [8,  9,  10, 11, 12, 13],
    "waist":  [15, 16, 17, 18, 19, 20],
    "neck":   [22, 23, 24, 25, 26, 27],
    "wrist":  [29, 30, 31, 32, 33, 34],
}

NODE_ORDER = ["ankle", "pocket", "waist", "neck", "wrist"]

ACTIVITY_NAMES = {
    0:  "Fall: hands",
    1:  "Fall: knees",
    2:  "Fall: backwards",
    3:  "Fall: sideways",
    4:  "Fall: chair",
    5:  "Walking",
    6:  "Standing",
    7:  "Sitting",
    8:  "Picking up",
    9:  "Jumping",
    10: "Lying",
}

FALL_CLASSES    = {0, 1, 2, 3, 4}
NON_FALL_CLASSES = {5, 6, 7, 8, 9, 10}

LABEL_COL   = 44
SUBJECT_COL = 43
TRIAL_COL   = 45


def load_gnn_windows(csv_path, verbose=True):
    """
    Load the UP-Fall consolidated CSV and return sliding-window samples.

    Returns
    -------
    X            : np.ndarray, shape (N, 5, 6, 36)
    y            : np.ndarray, shape (N,), int, values 0-10
    subject_ids  : np.ndarray, shape (N,), int
    """
    if verbose:
        print(f"Loading {csv_path} ...")

    df = pd.read_csv(csv_path, skiprows=2, header=None, low_memory=False)

    if verbose:
        print(f"  Raw rows: {len(df)}  |  Columns: {len(df.columns)}")

    sensor_cols = sum(SENSOR_MAP.values(), [])
    needed_cols = sensor_cols + [LABEL_COL, SUBJECT_COL, TRIAL_COL]

    for col in needed_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=needed_cols).reset_index(drop=True)
    df = df[df[LABEL_COL] != 20].reset_index(drop=True)
    df[LABEL_COL] = df[LABEL_COL].astype(int) - 1

    if verbose:
        print(f"  After cleaning: {len(df)} rows")
        counts = df[LABEL_COL].value_counts().sort_index()
        for idx, cnt in counts.items():
            print(f"    Class {idx:2d} ({ACTIVITY_NAMES[idx]:>20s}): {cnt:6d} samples")

    X_list, y_list, subj_list = [], [], []
    groups = df.groupby([SUBJECT_COL, TRIAL_COL, LABEL_COL])

    for (subj, trial, activity), group in groups:
        group = group.reset_index(drop=True)
        n = len(group)

        if n < WINDOW_SIZE:
            continue

        for start in range(0, n - WINDOW_SIZE + 1, STEP):
            window = group.iloc[start:start + WINDOW_SIZE]
            tags   = window[LABEL_COL].values
            label = int(np.bincount(tags.astype(int)).argmax())

            nodes = []
            for node_name in NODE_ORDER:
                cols = SENSOR_MAP[node_name]
                node_data = window[cols].values.T
                nodes.append(node_data)

            X_list.append(np.stack(nodes, axis=0))
            y_list.append(label)
            subj_list.append(int(subj))

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list,  dtype=np.int64)
    subject_ids = np.array(subj_list, dtype=np.int64)

    if verbose:
        print(f"\nFinal dataset: {X.shape[0]} windows, shape {X.shape}")
        counts = np.bincount(y, minlength=11)
        fall_total = counts[:5].sum()
        adl_total  = counts[5:].sum()
        print(f"  Falls (classes 0-4):  {fall_total:5d} windows")
        print(f"  ADLs  (classes 5-10): {adl_total:5d} windows")
        print(f"  Subjects: {np.unique(subject_ids).tolist()}")

    return X, y, subject_ids


if __name__ == "__main__":
    X, y, subject_ids = load_gnn_windows("data.csv")
    print("\nX shape:", X.shape)
    print("y unique labels:", np.unique(y))
    print("Subject IDs:", np.unique(subject_ids))