import json
import torch
import pandas as pd

from backend.app.config import (
    CKPT_PATH,
    CLASSES_PATH,
    CENTROID_PATH,
    TEMPERATURE_PATH,
    DEVICE
)
from backend.app.ml.model import KannadaClassifier


def load_classes(path):
    with open(path, "r", encoding="utf-8") as file:
        classes = json.load(file)

    if not isinstance(classes, list) or not classes:
        raise ValueError("Class JSON must contain a non-empty list.")

    return classes


def load_temperature(path):
    if not path.exists():
        print("Temperature file not found. Using temperature = 1.0")
        return 1.0

    saved = torch.load(path, map_location="cpu")

    if isinstance(saved, dict):
        return float(saved.get("temperature", 1.0))

    return float(saved)


def load_centroid_neighbors(path):
    if not path.exists():
        print("Centroid-neighbor CSV not found. Continuing without hints.")
        return {}

    dataframe = pd.read_csv(path, dtype=str).fillna("")
    lookup = {}

    for _, row in dataframe.iterrows():
        class_name = str(row.iloc[0]).strip()

        neighbors = [
            str(value).strip()
            for value in row.iloc[1:].tolist()
            if str(value).strip()
        ]

        lookup[class_name] = neighbors

    return lookup


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get(
            "model_state_dict",
            checkpoint.get("state_dict", checkpoint)
        )
    else:
        state_dict = checkpoint

    cleaned_state_dict = {
        key.replace("module.", ""): value
        for key, value in state_dict.items()
    }

    missing_keys, unexpected_keys = model.load_state_dict(
        cleaned_state_dict,
        strict=False
    )

    print("Missing keys:", len(missing_keys))
    print("Unexpected keys:", len(unexpected_keys))

    critical_prefixes = ("fc_embed.", "bn.", "classifier.")

    critical_missing = [
        key for key in missing_keys
        if key.startswith(critical_prefixes)
    ]

    if critical_missing:
        raise RuntimeError(
            "Checkpoint does not match the V3 custom head: "
            f"{critical_missing}"
        )

    return model


def load_inference_assets():
    if not CKPT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint missing: {CKPT_PATH}"
        )

    if not CLASSES_PATH.exists():
        raise FileNotFoundError(
            f"Class JSON missing: {CLASSES_PATH}"
        )

    all_classes = load_classes(CLASSES_PATH)

    model = KannadaClassifier(
        num_classes=len(all_classes)
    )

    model = load_checkpoint(model, CKPT_PATH)
    model = model.to(DEVICE)
    model.eval()

    assets = {
        "model": model,
        "classes": all_classes,
        "temperature": load_temperature(TEMPERATURE_PATH),
        "centroid_neighbors": load_centroid_neighbors(CENTROID_PATH)
    }

    print(f"Loaded {len(all_classes)} classes on {DEVICE}")
    print(f"Temperature: {assets['temperature']}")

    return assets