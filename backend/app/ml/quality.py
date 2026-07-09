import cv2
import numpy as np


def calculate_input_quality(image_pil):
    image_rgb = np.array(image_pil.convert("RGB"))
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_brightness = float(gray.mean())
    ink_ratio = float((gray < 180).mean())

    height, width = gray.shape

    return {
        "blur_score": blur_score,
        "mean_brightness": mean_brightness,
        "ink_ratio": ink_ratio,
        "aspect_ratio": float(width / max(height, 1))
    }