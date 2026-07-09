import pandas as pd

from backend.app.config import TELEMETRY_CSV


TELEMETRY_COLUMNS = [
    "request_id",
    "timestamp_utc",
    "model_version",
    "device",
    "predicted_label",
    "top1_confidence",
    "top1_top2_margin",
    "review_required",
    "decision",
    "decision_reason",
    "latency_ms",
    "blur_score",
    "mean_brightness",
    "ink_ratio",
    "aspect_ratio"
]


def initialize_telemetry():
    if not TELEMETRY_CSV.exists():
        pd.DataFrame(
            columns=TELEMETRY_COLUMNS
        ).to_csv(
            TELEMETRY_CSV,
            index=False,
            encoding="utf-8"
        )


def append_telemetry(record):
    initialize_telemetry()

    pd.DataFrame([record]).to_csv(
        TELEMETRY_CSV,
        mode="a",
        header=False,
        index=False,
        encoding="utf-8"
    )