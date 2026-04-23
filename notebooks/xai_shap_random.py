# =========================
# XAI: SHAP on RANDOM-SPLIT model (the "high performance" one)
# =========================
"""
xai_shap_random.py — SHAP on the random-split BiLSTM model.

Prerequisite: train_lstm_random.py must be run FIRST in the same Python
session (e.g., same Jupyter kernel or Colab runtime). This script re-uses
the trained model and normalised data arrays from that session rather than
retraining, which would take approximately 20 minutes.

If running standalone, retrain by running train_lstm_random.py and then
this script within the same kernel. In Jupyter/Colab, this means running
the training notebook cell before the SHAP notebook cell.
"""

import os, json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import shap

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR_RND = os.path.join(PROJECT_ROOT, "prepared_upfall_random")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "outputs", "xai")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Try to get feature names (fallback if metadata not present)
try:
    with open(os.path.join(DATA_DIR_RND, "metadata.json")) as f:
        feature_names = json.load(f)["feature_names"]
except Exception:
    with open(os.path.join(PROJECT_ROOT, "prepared_upfall", "metadata.json")) as f:
        feature_names = json.load(f)["feature_names"]

# Try to get activity labels (for the titles)
activities_rnd = None
for p in ["activities.npy", "activity_ids.npy"]:
    if os.path.exists(os.path.join(DATA_DIR_RND, p)):
        activities_rnd = np.load(os.path.join(DATA_DIR_RND, p))
        break

# Need to know which original windows went into the test set
# This assumes your random-split script saved train/test indices; if not, we'll warn.
try:
    split_info = np.load(os.path.join(DATA_DIR_RND, "split_indices.npz"))
    test_idx = split_info["test_idx"]
except Exception:
    print("No split_indices.npz found — activity labels on titles will be unavailable.")
    test_idx = None

# =========================
# Build predictions dataframe for test set
# =========================
preds_rnd = (test_probs > threshold).astype(int)
preds_df = pd.DataFrame({
    "local_idx": np.arange(len(y_te)),
    "y_true":    y_te.astype(int),
    "y_pred":    preds_rnd,
    "prob":      test_probs,
})
if test_idx is not None and activities_rnd is not None:
    preds_df["activity"] = activities_rnd[test_idx]
else:
    preds_df["activity"] = -1  # unknown

# =========================
# Set up SHAP (GradientExplainer, cuDNN off)
# =========================
torch.backends.cudnn.enabled = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rng = np.random.RandomState(42)
bg_idx = rng.choice(len(X_tr_norm), 100, replace=False)
background = torch.tensor(X_tr_norm[bg_idx], dtype=torch.float32).to(DEVICE)

model.eval()
explainer = shap.GradientExplainer(model, background)

# =========================
# Pick 4 windows: correct fall, missed fall, FP, correct reject
# =========================
def pick_one(df):
    return None if len(df) == 0 else int(df.iloc[0]["local_idx"])

candidates = {
    "correct_fall":   pick_one(preds_df[(preds_df.y_true==1) & (preds_df.y_pred==1)]),
    "missed_fall":    pick_one(preds_df[(preds_df.y_true==1) & (preds_df.y_pred==0)]),
    "false_positive": pick_one(preds_df[(preds_df.y_true==0) & (preds_df.y_pred==1)]),
    "correct_reject": pick_one(preds_df[(preds_df.y_true==0) & (preds_df.y_pred==0)]),
}

ACTIVITY_NAMES = {
    1:"fall/hands", 2:"fall/knees", 3:"fall/back", 4:"fall/sideways",
    5:"fall/sitting", 6:"walking", 7:"standing", 8:"sitting",
    9:"picking up", 10:"jumping", 11:"lying", -1:"unknown"
}

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

for i, (tag, local_idx) in enumerate(candidates.items()):
    if local_idx is None:
        axes[i].set_title(f"{tag}: no example (model too accurate?)")
        axes[i].axis("off")
        continue

    x_exp = torch.tensor(X_te_norm[local_idx:local_idx+1], dtype=torch.float32).to(DEVICE)
    shap_values = explainer.shap_values(x_exp, nsamples=200)
    sv = np.array(shap_values).squeeze()

    row = preds_df.iloc[local_idx]
    act = int(row["activity"])

    im = axes[i].imshow(sv.T, aspect="auto", cmap="RdBu_r",
                        vmin=-np.abs(sv).max(), vmax=np.abs(sv).max())
    axes[i].set_title(f"{tag}\nlocal_w={local_idx} activity={ACTIVITY_NAMES.get(act, act)} "
                      f"true={int(row.y_true)} pred={int(row.y_pred)} prob={row.prob:.2f}",
                      fontsize=10)
    axes[i].set_xlabel("Time step")
    axes[i].set_ylabel("Channel")
    axes[i].set_yticks(range(0, 30, 3))
    axes[i].set_yticklabels([feature_names[c] for c in range(0, 30, 3)], fontsize=7)
    plt.colorbar(im, ax=axes[i], label="SHAP")

plt.suptitle("SHAP on RANDOM-SPLIT model (literature protocol)", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "shap_random_split_four_cases.png"), dpi=120)
plt.show()
print(f"Saved → {os.path.join(OUTPUT_DIR, 'shap_random_split_four_cases.png')}")

# =========================
# Summary
# =========================
print("\nPrediction breakdown on test set:")
print(f"  Total test windows: {len(preds_df)}")
print(f"  True falls:        {int((preds_df.y_true==1).sum())}")
print(f"  Correct falls:     {int(((preds_df.y_true==1)&(preds_df.y_pred==1)).sum())}")
print(f"  Missed falls:      {int(((preds_df.y_true==1)&(preds_df.y_pred==0)).sum())}")
print(f"  False positives:   {int(((preds_df.y_true==0)&(preds_df.y_pred==1)).sum())}")
print(f"  Correct rejects:   {int(((preds_df.y_true==0)&(preds_df.y_pred==0)).sum())}")