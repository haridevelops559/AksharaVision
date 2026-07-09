import numpy as np

from backend.app.config import (
    MIN_BLUR_SCORE,
    MIN_INK_RATIO,
    MAX_INK_RATIO,
    REVIEW_CONFIDENCE_THRESHOLD,
    REVIEW_MARGIN_THRESHOLD
)


def route_prediction(probabilities, quality):
    sorted_indices = np.argsort(-probabilities)

    top1_confidence = float(probabilities[sorted_indices[0]])
    top2_confidence = float(probabilities[sorted_indices[1]])
    margin = top1_confidence - top2_confidence

    retake_reasons = []
    review_reasons = []

    if quality["blur_score"] < MIN_BLUR_SCORE:
        retake_reasons.append("image may be blurry")

    if quality["ink_ratio"] < MIN_INK_RATIO:
        retake_reasons.append("too little visible handwriting")

    if quality["ink_ratio"] > MAX_INK_RATIO:
        retake_reasons.append(
            "image may be overly dark or tightly cropped"
        )

    if retake_reasons:
        return {
            "decision": "RETAKE IMAGE",
            "review_required": True,
            "reason": "; ".join(retake_reasons),
            "top1_confidence": top1_confidence,
            "margin": margin
        }

    if top1_confidence < REVIEW_CONFIDENCE_THRESHOLD:
        review_reasons.append("low calibrated confidence")

    if margin < REVIEW_MARGIN_THRESHOLD:
        review_reasons.append(
            "ambiguous Top-1 / Top-2 prediction"
        )

    if review_reasons:
        return {
            "decision": "REVIEW REQUIRED",
            "review_required": True,
            "reason": "; ".join(review_reasons),
            "top1_confidence": top1_confidence,
            "margin": margin
        }

    return {
        "decision": "ACCEPT",
        "review_required": False,
        "reason": "confidence, margin, and quality checks passed",
        "top1_confidence": top1_confidence,
        "margin": margin
    }