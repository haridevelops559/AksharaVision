import cv2
from PIL import Image
from inference import predict_image
import os

def progressive_word_segmentation(image_path, max_splits=5):
    img = cv2.imread(image_path)
    h, w, _ = img.shape
    summary = ""

    for splits in range(2, max_splits+1):
        seg_w = w // splits
        chars = []

        for i in range(splits):
            x1, x2 = i*seg_w, w if i == splits-1 else (i+1)*seg_w
            crop = img[:, x1:x2]
            temp = f"_seg_{i}.png"
            cv2.imwrite(temp, crop)
            preds = predict_image(temp)
            chars.append(preds[0][1])

        summary += f"Split {splits}: {''.join(chars)}\n"

    return summary
