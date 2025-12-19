# AksharaVision — Kannada Handwritten Character Recognition

AksharaVision is an end-to-end OCR system for Kannada handwritten characters,
built using a Swin Transformer backbone with centroid-refined inference.

## Key Features
- 113 Kannada character classes
- Swin Transformer with embedding head
- Centroid-based refinement for hard samples
- Progressive word segmentation
- Confidence visualization
- Human-in-the-loop feedback system
- Gradio-based interactive UI

## Model Performance
- Accuracy: 99.89%
- Macro F1-score: 0.9989
- Expected Calibration Error (ECE): 0.001

## Pipeline Overview
1. Image preprocessing and normalization
2. Swin Transformer feature extraction
3. Embedding projection + classifier
4. Softmax prediction
5. Centroid similarity refinement
6. Progressive word segmentation
7. Feedback-driven data logging

## How to Run
```bash
pip install -r requirements.txt
python app.py
