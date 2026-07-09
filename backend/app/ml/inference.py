import time
import uuid
import datetime

import numpy as np
import torch

from backend.app.config import DEVICE, MODEL_VERSION
from backend.app.ml.preprocessing import inference_transform
from backend.app.ml.quality import calculate_input_quality
from backend.app.ml.routing import route_prediction
from backend.app.services.telemetry import append_telemetry


def get_display_character(label):
    return label.split("_", 1)[1] if "_" in label else label


def predict_character(image_pil, assets, topk=3, log=True):
    model = assets["model"]
    all_classes = assets["classes"]
    temperature = assets["temperature"]
    centroid_neighbors = assets["centroid_neighbors"]

    image = image_pil.convert("RGB")
    quality = calculate_input_quality(image)

    input_tensor = inference_transform(image).unsqueeze(0).to(DEVICE)

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    with torch.no_grad():
        logits, _ = model(input_tensor)

        probabilities = torch.softmax(
            logits / temperature,
            dim=1
        ).cpu().numpy()[0]

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    latency_ms = (time.perf_counter() - start_time) * 1000

    top_indices = np.argsort(-probabilities)[:topk]

    predictions = []

    for class_index in top_indices:
        label = all_classes[int(class_index)]

        predictions.append({
            "class_index": int(class_index),
            "label": label,
            "character": get_display_character(label),
            "probability": float(probabilities[class_index])
        })

    decision = route_prediction(probabilities, quality)
    request_id = str(uuid.uuid4())

    similar_class_hints = [
        hint for hint in centroid_neighbors.get(
            predictions[0]["label"],
            []
        )
        if hint in all_classes
    ][:3]

    result = {
        "request_id": request_id,
        "predictions": predictions,
        "decision": decision,
        "quality": quality,
        "latency_ms": latency_ms,
        "similar_class_hints": similar_class_hints
    }

    if log:
        append_telemetry({
            "request_id": request_id,
            "timestamp_utc": datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(),
            "model_version": MODEL_VERSION,
            "device": DEVICE,
            "predicted_label": predictions[0]["label"],
            "top1_confidence": decision["top1_confidence"],
            "top1_top2_margin": decision["margin"],
            "review_required": decision["review_required"],
            "decision": decision["decision"],
            "decision_reason": decision["reason"],
            "latency_ms": latency_ms,
            "blur_score": quality["blur_score"],
            "mean_brightness": quality["mean_brightness"],
            "ink_ratio": quality["ink_ratio"],
            "aspect_ratio": quality["aspect_ratio"]
        })

    return result