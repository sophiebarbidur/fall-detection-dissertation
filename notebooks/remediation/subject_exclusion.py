# =========================
# S10 exclusion experiment
# =========================
import os, json
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "outputs", "diagnostics")

preds_df = pd.read_csv(os.path.join(OUTPUT_DIR, "loso_per_window_predictions.csv"))

with open(os.path.join(PROJECT_ROOT, "outputs", "lstm", "lstm_5.6s_results.json")) as f:
    baseline = json.load(f)

# =========================
# All 15 subjects
# =========================
y_all  = preds_df["y_true"].values
yp_all = preds_df["y_pred"].values

f1_all  = [r["f1"]  for r in baseline["fold_results"]]
acc_all = [r["accuracy"] for r in baseline["fold_results"]]

print("="*60)
print("ALL 15 SUBJECTS (original)")
print("="*60)
print(f"  Pooled accuracy : {accuracy_score(y_all, yp_all):.4f}")
print(f"  Pooled F1       : {f1_score(y_all, yp_all, zero_division=0):.4f}")
print(f"  Pooled precision: {precision_score(y_all, yp_all, zero_division=0):.4f}")
print(f"  Pooled recall   : {recall_score(y_all, yp_all, zero_division=0):.4f}")
print(f"  Mean-of-folds F1: {np.mean(f1_all):.4f}")
print(f"  Std-of-folds F1 : {np.std(f1_all):.4f}")
print(f"  Confusion matrix:\n{confusion_matrix(y_all, yp_all)}")

# =========================
# Exclude S10
# =========================
sub14   = preds_df[preds_df["subject"] != 10]
folds14 = [r for r in baseline["fold_results"] if r["test_subject"] != 10]

y_14  = sub14["y_true"].values
yp_14 = sub14["y_pred"].values

print("\n" + "="*60)
print("EXCLUDING S10 (sensor artifact)")
print("="*60)
print(f"  Windows remaining: {len(sub14)} / {len(preds_df)}")
print(f"  Pooled accuracy : {accuracy_score(y_14, yp_14):.4f}")
print(f"  Pooled F1       : {f1_score(y_14, yp_14, zero_division=0):.4f}")
print(f"  Pooled precision: {precision_score(y_14, yp_14, zero_division=0):.4f}")
print(f"  Pooled recall   : {recall_score(y_14, yp_14, zero_division=0):.4f}")
print(f"  Mean-of-folds F1: {np.mean([r['f1'] for r in folds14]):.4f}")
print(f"  Std-of-folds F1 : {np.std([r['f1'] for r in folds14]):.4f}")
print(f"  Confusion matrix:\n{confusion_matrix(y_14, yp_14)}")

# =========================
# Exclude S10 + S13
# =========================
sub13   = preds_df[~preds_df["subject"].isin([10, 13])]
folds13 = [r for r in baseline["fold_results"] if r["test_subject"] not in [10, 13]]

y_13  = sub13["y_true"].values
yp_13 = sub13["y_pred"].values

print("\n" + "="*60)
print("EXCLUDING S10 + S13 (sensor artifact + non-compliance)")
print("="*60)
print(f"  Windows remaining: {len(sub13)} / {len(preds_df)}")
print(f"  Pooled accuracy : {accuracy_score(y_13, yp_13):.4f}")
print(f"  Pooled F1       : {f1_score(y_13, yp_13, zero_division=0):.4f}")
print(f"  Pooled precision: {precision_score(y_13, yp_13, zero_division=0):.4f}")
print(f"  Pooled recall   : {recall_score(y_13, yp_13, zero_division=0):.4f}")
print(f"  Mean-of-folds F1: {np.mean([r['f1'] for r in folds13]):.4f}")
print(f"  Std-of-folds F1 : {np.std([r['f1'] for r in folds13]):.4f}")
print(f"  Confusion matrix:\n{confusion_matrix(y_13, yp_13)}")

# =========================
# Summary table
# =========================
print("\n" + "="*60)
print("SUMMARY TABLE")
print("="*60)
print(f"{'Configuration':<30} {'Mean F1':>9} {'Std F1':>9} {'Pooled F1':>11}")
print("-"*62)
print(f"{'All 15 subjects':<30} "
      f"{np.mean(f1_all):>9.4f} "
      f"{np.std(f1_all):>9.4f} "
      f"{f1_score(y_all, yp_all, zero_division=0):>11.4f}")
print(f"{'Exclude S10':<30} "
      f"{np.mean([r['f1'] for r in folds14]):>9.4f} "
      f"{np.std([r['f1'] for r in folds14]):>9.4f} "
      f"{f1_score(y_14, yp_14, zero_division=0):>11.4f}")
print(f"{'Exclude S10 + S13':<30} "
      f"{np.mean([r['f1'] for r in folds13]):>9.4f} "
      f"{np.std([r['f1'] for r in folds13]):>9.4f} "
      f"{f1_score(y_13, yp_13, zero_division=0):>11.4f}")