from pathlib import Path
import json
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import DIAGNOSTICS_DIR

LOGITS_PATH = DIAGNOSTICS_DIR / "evaluation_logits.npy"
OUTPUT_PATH = DIAGNOSTICS_DIR / "deployment_drift_baseline.json"

if not LOGITS_PATH.exists():
    raise FileNotFoundError(f"Missing evaluation logits: {LOGITS_PATH}")

logits = np.load(LOGITS_PATH)

# NumPy softmax, stable for large logits.
shifted_logits = logits - logits.max(axis=1, keepdims=True)
probabilities = np.exp(shifted_logits)
probabilities /= probabilities.sum(axis=1, keepdims=True)

sorted_probabilities = np.sort(probabilities, axis=1)
top1_confidence = sorted_probabilities[:, -1]
top1_top2_margin = (
    sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
)

baseline = {
    "model_version": "swin-v3-release",
    "baseline_scope": (
        "Held-out evaluation output distribution. Image-quality feature "
        "baselines are not included because they were not exported from Colab."
    ),
    "reference_samples": {
        "top1_confidence": top1_confidence.astype(float).tolist(),
        "top1_top2_margin": top1_top2_margin.astype(float).tolist(),
    },
}

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT_PATH.open("w", encoding="utf-8") as file:
    json.dump(baseline, file, indent=2)

print(f"Saved baseline: {OUTPUT_PATH}")
print(f"Reference samples: {len(top1_confidence)}")
print(f"Mean confidence: {top1_confidence.mean():.6f}")
print(f"Mean margin: {top1_top2_margin.mean():.6f}")