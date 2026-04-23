# =========================
# LLM narrative generation 
# =========================
!pip -q install anthropic shap

import os, json
import numpy as np
import pandas as pd
import torch
import shap
from anthropic import Anthropic


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR_RND = os.path.join(PROJECT_ROOT, "prepared_upfall")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "outputs", "xai")

# ==================================================================
# 1. Put ANTHROPIC API key here
# ==================================================================
# IMPORTANT: Set ANTHROPIC_API_KEY environment variable before running.
# Get a key from console.anthropic.com → API Keys → Create Key.
# Costs approximately £0.01 per narrative; £0.04 total for the 4 cases.

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if API_KEY is None:
    raise EnvironmentError(
        "ANTHROPIC_API_KEY environment variable not set. "
        "Run: export ANTHROPIC_API_KEY='your-key-here' before executing this script."
    )


# =========================
# Load everything (assumes model from Cell 1 is in memory)
# =========================
X          = np.load(os.path.join(DATA_DIR, "X_windows.npy"))
activities = np.load(os.path.join(DATA_DIR, "activities.npy"))
groups     = np.load(os.path.join(DATA_DIR, "groups.npy"))
with open(os.path.join(DATA_DIR, "metadata.json")) as f:
    feature_names = json.load(f)["feature_names"]
preds_df = pd.read_csv(os.path.join(PROJECT_ROOT, "outputs", "diagnostics",
                                    "loso_per_window_predictions.csv"))

attn_data = np.load(os.path.join(OUTPUT_DIR, "attention_loso_s13.npz"))

ACTIVITY_NAMES = {
    1:"falling forward using hands", 2:"falling forward using knees",
    3:"falling backwards",            4:"falling sideways",
    5:"falling from sitting in a chair", 6:"walking",
    7:"standing",                      8:"sitting",
    9:"picking up an object",          10:"jumping", 11:"lying",
}

SENSOR_LOCATIONS = {
    "ankle":  "ankle",   "pocket": "trouser pocket",
    "belt":   "waist",   "neck":   "neck",
    "wrist":  "wrist",
}

def humanise_channel(channel_name):
    """Convert 'belt_acc_x' → 'waist accelerometer (x-axis)'."""
    parts = channel_name.split("_")
    loc   = SENSOR_LOCATIONS.get(parts[0], parts[0])
    sensor = "accelerometer" if parts[1] == "acc" else "gyroscope"
    return f"{loc} {sensor} ({parts[2]}-axis)"

# =========================
# SHAP setup (model should still be in memory from Cell 1)
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = False

rng = np.random.RandomState(42)
bg_idx = rng.choice(len(X_tr_norm), 100, replace=False)
background = torch.tensor(X_tr_norm[bg_idx], dtype=torch.float32).to(DEVICE)

model.eval()
explainer = shap.GradientExplainer(model, background)

# =========================
# Evidence extraction
# =========================
def extract_evidence(global_idx, tag):
    local_idx = global_to_local[int(global_idx)]

    # Raw signal
    w = X[int(global_idx)]
    motion_energy = float(np.mean(np.std(w, axis=0)))

    # SHAP
    x_tensor = torch.tensor(X_te_norm[local_idx:local_idx+1],
                            dtype=torch.float32).to(DEVICE)
    sv = np.array(explainer.shap_values(x_tensor, nsamples=200)).squeeze()  # (T, C)
    channel_contrib = sv.sum(axis=0)
    top_ch = np.argsort(np.abs(channel_contrib))[-3:][::-1]
    top_channels = [
        {
            "sensor": humanise_channel(feature_names[int(c)]),
            "contribution": round(float(channel_contrib[c]), 3),
            "pushes_toward": "fall" if channel_contrib[c] > 0 else "non-fall"
        }
        for c in top_ch
    ]

    # Attention
    attn = attn_data[f"{tag}_attn"]
    top_t = np.argsort(attn)[-3:][::-1]
    top_timesteps = [
        {
            "timestep": int(t),
            "time_in_window_seconds": round(int(t) / 18.0, 2),
            "attention_weight": round(float(attn[t]), 4)
        }
        for t in top_t
    ]

    row = preds_df[preds_df.window == int(global_idx)].iloc[0]

    return {
        "window_id": int(global_idx),
        "subject_id": 13,
        "true_activity": ACTIVITY_NAMES.get(int(activities[int(global_idx)]), "unknown"),
        "ground_truth_label": "fall" if int(row.y_true) == 1 else "non-fall",
        "model_prediction": "fall" if int(row.y_pred) == 1 else "non-fall",
        "model_confidence": round(float(row.prob), 3),
        "prediction_correct": bool(row.y_true == row.y_pred),
        "motion_energy_in_window": round(motion_energy, 3),
        "top_contributing_sensors": top_channels,
        "top_attention_timesteps": top_timesteps,
    }

# =========================
# LLM prompt
# =========================
client = Anthropic(api_key=API_KEY)

SYSTEM_PROMPT = """You are a clinical decision-support assistant explaining outputs
from a wearable-sensor fall-detection AI system. Your audience is non-technical:
carers, nurses, family members, and elderly users themselves.

You receive structured evidence about a single detection event and write a short,
clear 4-6 sentence explanation.

Your explanations must:
1. State plainly what the system predicted and how confident it was.
2. Describe in everyday language which sensor readings drove the decision. Refer
   to sensors by body location (ankle, waist, wrist, neck, trouser pocket) rather
   than by technical channel names.
3. Describe when in the measurement window the key evidence occurred, using
   phrases like "early in the 5.6-second measurement" or "around the 3-second
   mark" rather than "timestep 54".
4. If the prediction was incorrect, offer a plausible explanation based only on
   the evidence provided. Do not speculate beyond the data.
5. Where appropriate, give a brief practical recommendation for carers or system
   designers.

Keep the tone factual, calm, and professional. Avoid jargon and avoid hedging
excessively. Do not invent details not in the evidence."""

USER_TEMPLATE = """The system produced the following detection event:

{evidence}

Please provide a 4-6 sentence clinician-readable explanation."""

def explain_with_llm(evidence):
    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": USER_TEMPLATE.format(evidence=json.dumps(evidence, indent=2))
        }],
    )
    return message.content[0].text

# =========================
# Generate narratives for the 4 cases
# =========================
s13_preds = preds_df[preds_df["subject"] == 13]
def pick_one(df):
    return None if len(df) == 0 else int(df.iloc[0]["window"])

cases = {
    "correct_fall":   pick_one(s13_preds[(s13_preds.y_true==1) & (s13_preds.y_pred==1)]),
    "missed_fall":    pick_one(s13_preds[(s13_preds.y_true==1) & (s13_preds.y_pred==0)]),
    "false_positive": pick_one(s13_preds[(s13_preds.y_true==0) & (s13_preds.y_pred==1)]),
    "correct_reject": pick_one(s13_preds[(s13_preds.y_true==0) & (s13_preds.y_pred==0)]),
}

# Need global_to_local mapping from Cell 1 (S13 test indices)
# Reconstruct if needed
s13_test_idx = np.where(groups == 13)[0]
global_to_local = {int(g): i for i, g in enumerate(s13_test_idx)}

narratives = {}
for tag, global_idx in cases.items():
    if global_idx is None: continue
    print(f"\n{'='*70}\n{tag.upper()}  (window {global_idx})\n{'='*70}")

    evidence = extract_evidence(global_idx, tag)
    print("\n[STRUCTURED EVIDENCE]")
    print(json.dumps(evidence, indent=2))

    narrative = explain_with_llm(evidence)
    print("\n[LLM EXPLANATION]")
    print(narrative)

    narratives[tag] = {"evidence": evidence, "narrative": narrative}

# Save
with open(os.path.join(OUTPUT_DIR, "llm_narratives_s13.json"), "w") as f:
    json.dump(narratives, f, indent=2)
print(f"\n\nSaved → {os.path.join(OUTPUT_DIR, 'llm_narratives_s13.json')}")