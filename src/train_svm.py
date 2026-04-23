"""
SVM baseline for binary fall detection on UP-Fall dataset.
Uses the shared prepared data from prepare_upfall_windows.py.
LOSO-CV for fair comparison with CNN, LSTM, ST-GCN.
Tests multiple C values in one run.
"""

import os
import json
import time
import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# =========================================================
# PATHS
# =========================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "prepared_upfall")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "svm")
os.makedirs(OUTPUT_DIR, exist_ok=True)

X_PATH = os.path.join(DATA_DIR, "X_windows.npy")
Y_PATH = os.path.join(DATA_DIR, "y.npy")
GROUPS_PATH = os.path.join(DATA_DIR, "groups.npy")
FOLDS_PATH = os.path.join(DATA_DIR, "loso_folds.json")

# =========================================================
# CONFIG
# =========================================================
NUM_SENSORS = 5
CHANNELS_PER_SENSOR = 6
C_VALUES = [0.1, 1.0, 10.0, 50.0, 100.0]

# =========================================================
# FEATURE EXTRACTION
# =========================================================
def extract_features(window):
    """Per-sensor statistical + frequency features from a (T, F) window."""
    features = []

    for sensor_idx in range(NUM_SENSORS):
        start = sensor_idx * CHANNELS_PER_SENSOR
        sensor = window[:, start:start + CHANNELS_PER_SENSOR]

        acc = sensor[:, 0:3]
        gyro = sensor[:, 3:6]
        acc_mag = np.linalg.norm(acc, axis=1)
        gyro_mag = np.linalg.norm(gyro, axis=1)

        for signal in [acc_mag, gyro_mag]:
            features.extend([
                signal.mean(), signal.std(), signal.max(), signal.min(),
                signal.max() - signal.min(),
                np.median(signal),
                np.percentile(signal, 25),
                np.percentile(signal, 75),
                np.sqrt(np.mean(signal ** 2)),
                np.sum(np.abs(np.diff(signal))),
            ])

        for ch in range(CHANNELS_PER_SENSOR):
            sig = sensor[:, ch]
            features.extend([
                sig.mean(), sig.std(), sig.max(), sig.min(),
                np.sqrt(np.mean(sig ** 2)),
            ])

        for i, j in [(0, 1), (0, 2), (1, 2)]:
            corr = np.corrcoef(acc[:, i], acc[:, j])[0, 1]
            features.append(corr if np.isfinite(corr) else 0.0)

    return features


def extract_all_features(X):
    print("Extracting SVM features...")
    X_feat = np.array([extract_features(w) for w in X], dtype=np.float32)
    print(f"  Feature matrix shape: {X_feat.shape}")
    return X_feat


# =========================================================
# RUN LOSO FOR ONE C VALUE
# =========================================================
def run_loso(X_features, y, folds, C_val):
    """Run full LOSO-CV for a single C value. Returns results dict."""
    fold_results = []
    all_y_true = []
    all_y_pred = []
    total_start = time.time()

    for i, fold in enumerate(folds, start=1):
        test_subject = fold["test_subject"]
        train_idx = np.array(fold["train_indices"])
        test_idx = np.array(fold["test_indices"])

        X_train, X_test = X_features[train_idx], X_features[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        fold_start = time.time()
        print(f"  Fold {i:>2}/{len(folds)} | Subj{test_subject:02d}", end="  ")

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(
                kernel="rbf",
                C=C_val,
                gamma="scale",
                class_weight="balanced",
                random_state=RANDOM_SEED,
            ))
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        fold_time = time.time() - fold_start

        print(f"acc={acc:.2%}  f1={f1:.3f}  ({fold_time:.1f}s)")

        fold_results.append({
            "test_subject": int(test_subject),
            "accuracy": float(acc),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1),
            "num_test_samples": int(len(test_idx)),
            "num_falls_test": int(np.sum(y_test == 1)),
            "time_seconds": round(fold_time, 2),
        })

        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())

    total_time = time.time() - total_start
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    accs = [r["accuracy"] for r in fold_results]
    f1s = [r["f1"] for r in fold_results]

    return {
        "C": C_val,
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs)),
        "mean_f1": float(np.mean(f1s)),
        "std_f1": float(np.std(f1s)),
        "pooled_accuracy": float(accuracy_score(all_y_true, all_y_pred)),
        "pooled_precision": float(precision_score(all_y_true, all_y_pred, zero_division=0)),
        "pooled_recall": float(recall_score(all_y_true, all_y_pred, zero_division=0)),
        "pooled_f1": float(f1_score(all_y_true, all_y_pred, zero_division=0)),
        "pooled_confusion_matrix": confusion_matrix(all_y_true, all_y_pred).tolist(),
        "classification_report": classification_report(
            all_y_true, all_y_pred,
            target_names=["Non-Fall", "Fall"], zero_division=0
        ),
        "total_time_seconds": round(total_time, 2),
        "fold_results": fold_results,
    }


# =========================================================
# MAIN
# =========================================================
def main():
    print("Loading prepared dataset...")
    X_windows = np.load(X_PATH)
    y = np.load(Y_PATH)
    groups = np.load(GROUPS_PATH)

    with open(FOLDS_PATH, "r") as f:
        folds = json.load(f)

    print(f"  X_windows shape: {X_windows.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Falls: {np.sum(y == 1)}, Non-falls: {np.sum(y == 0)}")
    print(f"  LOSO folds: {len(folds)}")

    X_features = extract_all_features(X_windows)

    # ----- Run all C values -----
    all_results = {}

    for C_val in C_VALUES:
        print(f"\n{'='*60}")
        print(f"  C = {C_val}")
        print(f"{'='*60}")
        all_results[str(C_val)] = run_loso(X_features, y, folds, C_val)

    # =====================================================
    # COMPARISON TABLE
    # =====================================================
    print(f"\n{'='*60}")
    print("COMPARISON ACROSS C VALUES")
    print(f"{'='*60}\n")

    print(f"{'C':>8}  {'MeanAcc':>8}  {'StdAcc':>8}  {'MeanF1':>8}  "
          f"{'PoolAcc':>8}  {'PoolF1':>8}  {'PoolPrec':>9}  {'PoolRec':>8}  {'Time':>6}")
    print("-" * 85)

    best_c = None
    best_f1 = -1

    for C_val in C_VALUES:
        r = all_results[str(C_val)]
        print(f"{C_val:>8.1f}  {r['mean_accuracy']:>7.2%}  {r['std_accuracy']:>7.2%}  "
              f"{r['mean_f1']:>8.3f}  {r['pooled_accuracy']:>7.2%}  "
              f"{r['pooled_f1']:>8.3f}  {r['pooled_precision']:>9.3f}  "
              f"{r['pooled_recall']:>7.3f}  {r['total_time_seconds']:>5.1f}s")

        if r["mean_f1"] > best_f1:
            best_f1 = r["mean_f1"]
            best_c = C_val

    print("-" * 85)
    print(f"\n>>> Best C = {best_c} (mean LOSO F1 = {best_f1:.3f})")

    # Print full report for best C
    best = all_results[str(best_c)]
    print(f"\n{'='*60}")
    print(f"BEST MODEL DETAILS (C={best_c})")
    print(f"{'='*60}\n")

    print(f"{'Subject':>8}  {'Acc':>8}  {'F1':>8}  {'Prec':>8}  {'Recall':>8}")
    print("-" * 48)
    for r in best["fold_results"]:
        print(f"{r['test_subject']:>8}  {r['accuracy']:>7.2%}  {r['f1']:>8.3f}  "
              f"{r['precision']:>8.3f}  {r['recall']:>8.3f}")
    print("-" * 48)
    print(f"{'Mean':>8}  {best['mean_accuracy']:>7.2%}  {best['mean_f1']:>8.3f}")
    print(f"{'Std':>8}  {best['std_accuracy']:>7.2%}  {best['std_f1']:>8.3f}")

    print(f"\n{best['classification_report']}")
    print("Confusion Matrix (rows=true, cols=predicted):")
    print(np.array(best["pooled_confusion_matrix"]))

    # =====================================================
    # SAVE EVERYTHING
    # =====================================================
    # Save all C results
    save_all = {
        "model": "SVM",
        "evaluation": "LOSO",
        "C_values_tested": C_VALUES,
        "best_C": best_c,
        "best_mean_f1": best_f1,
        "num_features": int(X_features.shape[1]),
        "data_shape": list(X_windows.shape),
        "results_per_C": all_results,
    }

    all_path = os.path.join(OUTPUT_DIR, "svm_c_search.json")
    with open(all_path, "w") as f:
        json.dump(save_all, f, indent=2)

    # Save best C separately (for easy comparison with CNN/LSTM later)
    best_save = {
        "model": "SVM",
        "evaluation": "LOSO",
        "svm_params": {
            "kernel": "rbf",
            "C": best_c,
            "gamma": "scale",
            "class_weight": "balanced",
        },
        "num_features": int(X_features.shape[1]),
        "num_folds": len(folds),
        **{k: best[k] for k in [
            "mean_accuracy", "std_accuracy", "mean_f1", "std_f1",
            "pooled_accuracy", "pooled_precision", "pooled_recall",
            "pooled_f1", "pooled_confusion_matrix", "total_time_seconds",
            "fold_results"
        ]},
        "data_shape": list(X_windows.shape),
    }

    best_path = os.path.join(OUTPUT_DIR, "svm_results.json")
    with open(best_path, "w") as f:
        json.dump(best_save, f, indent=2)

    print(f"\nAll C results saved to: {all_path}")
    print(f"Best model saved to:    {best_path}")


if __name__ == "__main__":
    main()