from pathlib import Path
import os
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
DIAGNOSTICS_DIR = PROJECT_ROOT / "diagnostics"
STORAGE_DIR = PROJECT_ROOT / "backend" / "app" / "storage"

STORAGE_DIR.mkdir(parents=True, exist_ok=True)

CKPT_PATH = ARTIFACT_DIR / "kannada_classifier_finetuned_full.pth"
CLASSES_PATH = ARTIFACT_DIR / "classes_list_113_kannada.json"
CENTROID_PATH = ARTIFACT_DIR / "centroid_top3_neighbors.csv"
TEMPERATURE_PATH = ARTIFACT_DIR / "temperature_calibration.pt"
DRIFT_BASELINE_PATH = ARTIFACT_DIR / "deployment_drift_baseline.json"

TELEMETRY_CSV = STORAGE_DIR / "telemetry.csv"
FEEDBACK_CSV = STORAGE_DIR / "feedback.csv"
FEEDBACK_DB_PATH = STORAGE_DIR / "feedback.json"

MODEL_VERSION = os.getenv("MODEL_VERSION", "swin-v3-release")

USE_CUDA = os.getenv("USE_CUDA", "false").lower() == "true"
DEVICE = "cuda" if USE_CUDA and torch.cuda.is_available() else "cpu"

REVIEW_CONFIDENCE_THRESHOLD = 0.85
REVIEW_MARGIN_THRESHOLD = 0.10

MIN_BLUR_SCORE = 20.0
MIN_INK_RATIO = 0.01
MAX_INK_RATIO = 0.85