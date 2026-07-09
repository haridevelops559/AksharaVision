from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tinydb import TinyDB

from backend.app.config import STORAGE_DIR

FEEDBACK_CSV = STORAGE_DIR / "feedback.csv"
FEEDBACK_DB_PATH = STORAGE_DIR / "feedback.json"

FEEDBACK_COLUMNS = [
    "timestamp_utc",
    "request_id",
    "top1_label",
    "top2_label",
    "top3_label",
    "correct_label",
    "feedback_type",
]


def initialize_feedback_storage() -> None:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    if not FEEDBACK_CSV.exists():
        with FEEDBACK_CSV.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=FEEDBACK_COLUMNS)
            writer.writeheader()


def save_feedback(
    *,
    request_id: str,
    predictions: list[dict[str, Any]],
    feedback_type: str,
    manual_label: str | None,
    valid_classes: list[str],
) -> dict[str, str]:
    if len(predictions) < 3:
        raise ValueError("Feedback requires three prediction records.")

    allowed_types = {
        "Top-1 correct",
        "Top-2 correct",
        "Top-3 correct",
        "Manual correction",
    }

    if feedback_type not in allowed_types:
        raise ValueError(
            "feedback_type must be one of: "
            "Top-1 correct, Top-2 correct, Top-3 correct, Manual correction."
        )

    if feedback_type == "Top-1 correct":
        correct_label = predictions[0]["label"]
    elif feedback_type == "Top-2 correct":
        correct_label = predictions[1]["label"]
    elif feedback_type == "Top-3 correct":
        correct_label = predictions[2]["label"]
    else:
        correct_label = (manual_label or "").strip()

        if not correct_label:
            raise ValueError(
                "manual_label is required when feedback_type is Manual correction."
            )

        if correct_label not in valid_classes:
            raise ValueError(
                "manual_label must exactly match one of the 113 model labels."
            )

    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "top1_label": predictions[0]["label"],
        "top2_label": predictions[1]["label"],
        "top3_label": predictions[2]["label"],
        "correct_label": correct_label,
        "feedback_type": feedback_type,
    }

    initialize_feedback_storage()

    with FEEDBACK_CSV.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FEEDBACK_COLUMNS)
        writer.writerow(record)

    database = TinyDB(FEEDBACK_DB_PATH)
    database.insert(record)
    database.close()

    return {
        "status": "saved",
        "message": (
            "Feedback saved for review. It is not used for automatic retraining."
        ),
        "request_id": request_id,
        "correct_label": correct_label,
    }