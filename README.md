# AksharaVision

> **Production-Oriented Kannada Handwritten Character Recognition using Swin Transformer, FastAPI, and React**

![Python](https://img.shields.io/badge/Python-3.13-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![React](https://img.shields.io/badge/React-Frontend-61DAFB)
![License](https://img.shields.io/badge/License-MIT-orange)

---

## Overview

AksharaVision is an end-to-end Optical Character Recognition (OCR) system for handwritten Kannada characters. The project demonstrates the complete applied machine learning lifecycle—from dataset construction and model development to deployment, monitoring, and human-in-the-loop feedback.

The deployed system recognizes **113 handwritten Kannada character classes** using a fine-tuned **Swin Transformer** and exposes production-style inference APIs through **FastAPI**.

### Key Highlights

- 113-class handwritten Kannada OCR
- Swin Transformer V3 classifier
- 56,500 curated training images
- Calibrated Top-3 predictions
- Image quality assessment
- Accept / Review / Retake decision routing
- Occlusion-based explainability
- Human feedback collection
- Deployment telemetry logging
- PSI-based drift monitoring
- Interactive Swagger API documentation
- React dashboard (In Progress)

---

# System Architecture

```text
                    Handwritten Image
                           │
                           ▼
                  Image Preprocessing
                           │
                           ▼
                 Swin Transformer V3
                           │
                           ▼
                Calibrated Top-3 Prediction
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
 Image Quality       Explainability      Feedback API
     Checks          (Occlusion Map)         │
        │                  │                 ▼
        └──────────────► Telemetry ◄─────────┘
                           │
                           ▼
                    PSI Drift Monitoring
```

---

# Repository Structure

```text
AksharaVision
│
├── backend/              # FastAPI backend
├── frontend/             # React frontend (ongoing)
├── artifacts/            # Model metadata
├── diagnostics/          # Evaluation artifacts
├── notebooks/            # Research & experimentation
├── scripts/              # Utility scripts
├── README.md
└── .gitignore
```

---

# Model Performance

| Metric | Value |
|---------|-------:|
| Classes | 113 |
| Dataset Size | 56,500 Images |
| Top-1 Accuracy | **99.89%** |
| Top-2 Accuracy | **99.93%** |
| Top-3 Accuracy | **99.95%** |
| Macro F1 | **0.9989** |
| Expected Calibration Error | **0.00065** |
| Brier Score | **0.00112** |
| Single Image Latency | **36 ms** |

---

# Technology Stack

### Machine Learning

- PyTorch
- Swin Transformer
- OpenCV
- Albumentations
- Scikit-learn

### Backend

- FastAPI
- Uvicorn
- Pydantic

### Deployment

- Telemetry Logging
- PSI Drift Monitoring
- Feedback Collection

### Frontend

- React
- Axios
- Vite

### Experiment Tracking

- Weights & Biases (W&B)

---

# Getting Started

## Clone Repository

```bash
git clone https://github.com/haridevelops559/AksharaVision.git

cd AksharaVision
```

---

## Install Dependencies

```bash
cd backend

pip install -r requirements.txt
```

---

## Model Checkpoint

The trained model checkpoint is **not included** because it exceeds GitHub's file size limit.

Place the checkpoint inside:

```text
artifacts/

kannada_classifier_finetuned_full.pth
```

The class mapping JSON should also be available inside the same folder.

---

# Running the Backend

```bash
cd backend

uvicorn app.main:app --reload
```

The API will be available at

```
http://127.0.0.1:8000
```

Swagger Documentation

```
http://127.0.0.1:8000/docs
```

---

# API Endpoints

## Health Check

```
GET /health
```

Example Response

```json
{
  "status": "ok",
  "service": "aksharavision-api"
}
```

---

## Character Prediction

```
POST /predict
```

Upload a handwritten Kannada character image.

Returns

- Top-3 predictions
- Confidence scores
- Image quality metrics
- Accept / Review / Retake decision
- Similar class hints

Example Response

```json
{
  "predictions": [
    {
      "label": "031_ತ",
      "probability": 0.9917
    }
  ],
  "decision": {
    "decision": "ACCEPT"
  },
  "quality": {
    "blur_score": 18.2
  }
}
```

---

## Explain Prediction

```
POST /explain
```

Generates an occlusion sensitivity heatmap showing image regions that most influence the model prediction.

---

## Submit Feedback

```
POST /feedback
```

Example Request

```json
{
    "request_id":"<prediction_request_id>",
    "correct_label":"031_ತ"
}
```

Feedback is stored for manual review and **is not automatically used for retraining**.

---

## Deployment Monitoring

```
GET /monitoring
```

Returns

- Request statistics
- Average latency
- Confidence distribution
- Review rate
- PSI drift analysis
- Telemetry summary

---

# Reliability & Monitoring

AksharaVision incorporates several production-oriented reliability features.

- Temperature-scaled confidence calibration
- Top-3 prediction confidence
- Image quality assessment
- Accept / Review / Retake routing
- Human feedback logging
- Anonymous inference telemetry
- PSI-based drift monitoring
- Privacy-aware deployment (uploaded images are not stored by default)

---

# Engineering Highlights

- Built a 56.5K image Kannada OCR dataset using OpenCV-based extraction.
- Benchmarked CNN and Vision Transformer architectures before selecting Swin Transformer V3.
- Improved reliability through temperature scaling and calibration metrics.
- Integrated explainability using occlusion sensitivity analysis.
- Designed FastAPI inference APIs with structured request validation.
- Added telemetry, feedback collection, and deployment monitoring.
- Implemented PSI-based drift detection for post-deployment data monitoring.

---

# Project Evolution

| Version | Description |
|----------|-------------|
| V1 | Dataset construction and model experimentation in Google Colab |
| V2 | Interactive Gradio OCR application |
| V3 | FastAPI backend with telemetry and monitoring |
| V4 | React frontend (currently under development) |

---

# Future Work

- Complete React dashboard
- Docker deployment
- ONNX Runtime inference
- Active Learning workflow
- Kubernetes deployment
- Mobile OCR application
- End-to-end word recognition
- Multilingual Indic OCR

---

# License

This project is released under the MIT License.

---

# Author

**Shrihari M V**

- LinkedIn: https://www.linkedin.com/in/shrihari-athreyas
- GitHub: https://github.com/haridevelops559

---

## Acknowledgements

This project was developed as part of my applied machine learning portfolio to explore computer vision, reliable deep learning, explainable AI, and production-oriented ML deployment practices.
