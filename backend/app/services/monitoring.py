from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from backend.app.config import DIAGNOSTICS_DIR
from backend.app.services.telemetry import TELEMETRY_CSV

DRIFT_BASELINE_PATH = DIAGNOSTICS_DIR / "deployment_drift_baseline.json"


def load_drift_baseline() -> dict | None:
    if not DRIFT_BASELINE_PATH.exists():
        return None

    with DRIFT_BASELINE_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def population_stability_index(
    reference: np.ndarray,
    current: np.ndarray,
    bins: int = 10,
) -> float:
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]

    if len(reference) == 0 or len(current) == 0:
        return float("nan")

    edges = np.histogram_bin_edges(reference, bins=bins)

    # Avoid a degenerate bin range when a metric is nearly constant.
    if len(np.unique(edges)) < 2:
        lower = float(reference.min()) - 1e-6
        upper = float(reference.max()) + 1e-6
        edges = np.linspace(lower, upper, bins + 1)

    reference_counts, _ = np.histogram(reference, bins=edges)
    current_counts, _ = np.histogram(current, bins=edges)

    reference_pct = reference_counts / max(reference_counts.sum(), 1)
    current_pct = current_counts / max(current_counts.sum(), 1)

    epsilon = 1e-6
    reference_pct = np.clip(reference_pct, epsilon, None)
    current_pct = np.clip(current_pct, epsilon, None)

    return float(
        np.sum(
            (current_pct - reference_pct)
            * np.log(current_pct / reference_pct)
        )
    )


def classify_psi(psi: float) -> str:
    if not np.isfinite(psi):
        return "baseline unavailable"
    if psi < 0.10:
        return "stable"
    if psi < 0.25:
        return "monitor"
    return "material shift"


def build_monitoring_report() -> dict:
    if not TELEMETRY_CSV.exists():
        return {
            "summary": {
                "total_requests": 0,
                "message": "No telemetry has been recorded yet."
            },
            "drift": [],
            "telemetry_available": False
        }

    telemetry = pd.read_csv(TELEMETRY_CSV)

    if telemetry.empty:
        return {
            "summary": {
                "total_requests": 0,
                "message": "No telemetry has been recorded yet."
            },
            "drift": [],
            "telemetry_available": True
        }

    telemetry["review_required"] = (
        telemetry["review_required"]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    numeric_columns = [
        "latency_ms",
        "top1_confidence",
        "top1_top2_margin",
        "blur_score",
        "mean_brightness",
        "ink_ratio",
        "aspect_ratio",
    ]

    for column in numeric_columns:
        if column in telemetry.columns:
            telemetry[column] = pd.to_numeric(
                telemetry[column],
                errors="coerce"
            )

    summary = {
        "total_requests": int(len(telemetry)),
        "review_required_rate": round(
            float(telemetry["review_required"].mean()),
            4
        ),
        "mean_latency_ms": round(
            float(telemetry["latency_ms"].mean()),
            2
        ),
        "p95_latency_ms": round(
            float(telemetry["latency_ms"].quantile(0.95)),
            2
        ),
        "mean_top1_confidence": round(
            float(telemetry["top1_confidence"].mean()),
            4
        ),
        "mean_top1_top2_margin": round(
            float(telemetry["top1_top2_margin"].mean()),
            4
        ),
        "mean_blur_score": round(
            float(telemetry["blur_score"].mean()),
            2
        ),
        "mean_brightness": round(
            float(telemetry["mean_brightness"].mean()),
            2
        ),
        "mean_ink_ratio": round(
            float(telemetry["ink_ratio"].mean()),
            4
        ),
        "model_versions": (
            telemetry["model_version"]
            .dropna()
            .astype(str)
            .value_counts()
            .to_dict()
        ),
    }

    baseline = load_drift_baseline()
    drift_rows = []

    if baseline is not None:
        reference_samples = baseline.get("reference_samples", {})

        for metric in [
            "top1_confidence",
            "top1_top2_margin",
            "blur_score",
            "mean_brightness",
            "ink_ratio",
        ]:
            if metric not in telemetry.columns:
                continue

            reference = reference_samples.get(metric, [])
            current = telemetry[metric].dropna().to_numpy()

            psi = population_stability_index(reference, current)

            drift_rows.append({
                "metric": metric,
                "psi": round(psi, 4) if np.isfinite(psi) else None,
                "status": classify_psi(psi),
                "reference_count": int(len(reference)),
                "current_count": int(len(current)),
            })

    return {
        "summary": summary,
        "drift": drift_rows,
        "telemetry_available": True,
        "baseline_available": baseline is not None,
        "telemetry_file": str(TELEMETRY_CSV),
        "drift_baseline_file": str(DRIFT_BASELINE_PATH),
        "privacy_note": (
            "Telemetry stores prediction metadata and input-quality signals; "
            "uploaded images are not stored by default."
        ),
    }