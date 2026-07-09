from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from backend.app.services.feedback import save_feedback

router = APIRouter(tags=["feedback"])


class PredictionItem(BaseModel):
    label: str = Field(..., examples=["031_ತ"])


class FeedbackRequest(BaseModel):
    request_id: str = Field(..., min_length=1)
    predictions: list[PredictionItem] = Field(..., min_length=3, max_length=3)
    feedback_type: Literal[
        "Top-1 correct",
        "Top-2 correct",
        "Top-3 correct",
        "Manual correction",
    ]
    manual_label: str | None = None


@router.post("/feedback")
def submit_feedback(payload: FeedbackRequest, request: Request):
    try:
        return save_feedback(
            request_id=payload.request_id,
            predictions=[
                prediction.model_dump()
                for prediction in payload.predictions
            ],
            feedback_type=payload.feedback_type,
            manual_label=payload.manual_label,
            valid_classes=request.app.state.assets["classes"],
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error