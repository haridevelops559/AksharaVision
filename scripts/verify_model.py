from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.ml.loader import load_inference_assets

assets = load_inference_assets()

print("\nModel verification passed.")
print("Class count:", len(assets["classes"]))
print("First five labels:", assets["classes"][:5])
print("Temperature:", assets["temperature"])