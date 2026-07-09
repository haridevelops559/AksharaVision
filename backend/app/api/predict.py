from io import BytesIO

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError

from backend.app.ml.inference import predict_character

router = APIRouter(tags=["prediction"])


@router.post("/predict")
async def predict(
    request: Request,
    file: UploadFile = File(...)
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail="Upload a PNG, JPG, JPEG, WEBP, or BMP image."
        )

    try:
        raw_bytes = await file.read()
        image = Image.open(BytesIO(raw_bytes)).convert("RGB")
    except UnidentifiedImageError as error:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file could not be read as an image."
        ) from error

    result = predict_character(
        image_pil=image,
        assets=request.app.state.assets,
        topk=3,
        log=True
    )

    return result