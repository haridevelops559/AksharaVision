````md
# AksharaVision  
## Robust Kannada Handwritten Character Recognition using Swin Transformer, Calibration, and Feedback-Driven Learning

> **AksharaVision** is an end-to-end Kannada handwritten character recognition system designed for robust, reliable, and deployable OCR. The project combines dataset curation, augmentation, CNN baselines, hierarchical Swin Transformers, focal loss, centroid refinement, Expected Calibration Error (ECE), robustness testing, progressive word segmentation, and a Gradio-based feedback interface.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Gradio](https://img.shields.io/badge/Gradio-Interactive%20UI-orange)
![OCR](https://img.shields.io/badge/Task-Kannada%20Handwritten%20OCR-green)
![Transformer](https://img.shields.io/badge/Model-Swin%20Transformer-purple)

---

## Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Key Contributions](#key-contributions)
- [Dataset Curation](#dataset-curation)
- [Dataset Description](#dataset-description)
- [Preprocessing Pipeline](#preprocessing-pipeline)
- [Albumentations Strategy](#albumentations-strategy)
- [Baseline Model Exploration](#baseline-model-exploration)
- [Why Swin Transformer](#why-swin-transformer)
- [Swin Transformer Architecture](#swin-transformer-architecture)
- [Swin V1, V2, and V3](#swin-v1-v2-and-v3)
- [Why Focal Loss Instead of Cross-Entropy](#why-focal-loss-instead-of-cross-entropy)
- [Centroid Refinement](#centroid-refinement)
- [Calibration and Expected Calibration Error](#calibration-and-expected-calibration-error)
- [Robustness and Stress Testing](#robustness-and-stress-testing)
- [Metrics Comparison](#metrics-comparison)
- [Gradio Deployment](#gradio-deployment)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Training](#training)
- [Inference](#inference)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Citation](#citation)

---

# Project Overview

Kannada handwritten OCR is a challenging computer vision problem because Kannada characters contain:

- Curved and loop-based strokes
- Vowel modifiers
- Compound characters
- Ligatures
- Similar-looking character pairs
- Large writer-to-writer variation
- Ink thickness variation
- Background noise
- Scan and camera distortions
- Uneven lighting
- Blur and cropping errors

Traditional OCR pipelines and simple CNNs often struggle to capture both local stroke details and global character structure.

AksharaVision addresses this challenge using a carefully curated dataset and a hierarchical Swin Transformer architecture.

The system is designed as a complete OCR pipeline:

```text
Handwritten Sheets
        ↓
PDF / Image Conversion
        ↓
Contour Detection and Character Cropping
        ↓
Dataset Cleaning and Balancing
        ↓
Preprocessing and Albumentations
        ↓
CNN Baselines and Transformer Experiments
        ↓
Swin Transformer V1 → V2 → V3
        ↓
Focal Loss + Centroid Refinement
        ↓
ECE Calibration + Robustness Testing
        ↓
Gradio OCR Interface
        ↓
Feedback Logging and Future Retraining
````

---

# Problem Statement

Existing Kannada OCR systems face several limitations:

1. Limited availability of large, clean, and standardized handwritten Kannada datasets.
2. High intra-class variation caused by different handwriting styles.
3. High inter-class similarity between visually related characters.
4. Difficulty in recognizing compound characters and ligatures.
5. Poor confidence reliability in many deep learning models.
6. Limited robustness to real-world distortions such as blur, dark ink, background noise, zoom, and cropping.
7. Lack of interactive feedback systems for continuous improvement.

The goal of this project is to develop a robust handwritten Kannada character recognition system that is accurate, calibrated, interpretable, and deployable.

---

# Key Contributions

The project makes the following contributions:

* Curated a balanced handwritten Kannada dataset containing **113 character classes**.
* Built an automated sheet-to-character extraction pipeline using OpenCV contour detection.
* Balanced the dataset to **500 images per class**, resulting in **56,500 images**.
* Explored multiple architectures:

  * ResNet + BiLSTM + Attention
  * EfficientNet + CBAM
  * Swin Transformer V1
  * Swin Transformer V2
  * Swin Transformer V3
* Replaced standard cross-entropy with **Focal Loss** to improve learning on difficult and visually similar classes.
* Added a **512-dimensional embedding head**.
* Introduced **centroid-based refinement** for stable predictions.
* Applied **temperature scaling** to improve confidence calibration.
* Evaluated reliability using **Expected Calibration Error (ECE)**.
* Performed robustness tests under ink darkening, background jitter, and zoom/crop distortion.
* Built a Gradio interface for:

  * Single-character OCR
  * Word-level OCR using progressive segmentation
  * User feedback collection
  * JSON and CSV feedback storage

---

# Dataset Curation

## Data Collection

The dataset was created using structured handwritten data collection sheets.

Each sheet contains predefined boxes where contributors write Kannada characters. This controlled layout simplifies downstream extraction and ensures that each handwritten sample is isolated.

The data collection workflow is:

```text
Character Sheet Design
        ↓
Handwritten Character Collection
        ↓
Scanning / Mobile Camera Capture
        ↓
PDF or Image Upload
        ↓
Image Conversion
        ↓
Contour Detection
        ↓
Grid Detection
        ↓
Character Cell Cropping
        ↓
Class-Wise Dataset Storage
```

---

## Automated Character Extraction

The extraction pipeline uses OpenCV-based image processing.

### Main steps

1. Convert scanned PDF sheets into PNG images.
2. Convert images to grayscale.
3. Apply thresholding and edge detection.
4. Detect contours corresponding to grid cells.
5. Sort contours in reading order.
6. Crop individual character boxes.
7. Save cropped samples into class-specific directories.
8. Remove corrupted or incorrectly cropped samples.
9. Validate labels manually where necessary.

Example class directory structure:

```text
dataset/
├── ka/
│   ├── ka_001.png
│   ├── ka_002.png
│   └── ...
├── kha/
│   ├── kha_001.png
│   ├── kha_002.png
│   └── ...
├── ga/
│   ├── ga_001.png
│   ├── ga_002.png
│   └── ...
└── ...
```

---

# Dataset Description

| Property             |                                Value |
| -------------------- | -----------------------------------: |
| Script               |                              Kannada |
| Task                 | Handwritten Character Classification |
| Number of classes    |                                  113 |
| Samples per class    |                                  500 |
| Total images         |                               56,500 |
| Input size           |                            224 × 224 |
| Split strategy       |                     Stratified Split |
| Training split       |                                  90% |
| Validation split     |                                  10% |
| Data format          |                            PNG / JPG |
| Dataset organization |                   Class-wise folders |

The dataset contains:

* Kannada vowels
* Kannada consonants
* Compound characters
* Ligature-like forms
* Handwritten variants from multiple contributors

---

# Preprocessing Pipeline

The preprocessing pipeline standardizes raw handwritten samples before training.

```text
Raw Image
    ↓
Grayscale Conversion
    ↓
Noise Removal
    ↓
Foreground Enhancement
    ↓
Resize to 224 × 224
    ↓
Normalization
    ↓
Tensor Conversion
    ↓
Model Input
```

## Preprocessing Steps

### 1. Grayscale Conversion

Images are converted to grayscale to reduce unnecessary color variation.

### 2. Denoising

Noise introduced by scanning, compression, camera capture, or paper texture is reduced.

### 3. Foreground Normalization

Character strokes are enhanced while reducing background influence.

### 4. Resizing

All images are resized to:

```text
224 × 224
```

This resolution is compatible with Swin Transformer and CNN backbones.

### 5. Normalization

Pixel values are normalized before being passed into the neural network.

---

# Albumentations Strategy

Albumentations is used to simulate realistic handwriting and scanning variations.

The goal is not only to increase dataset size but also to improve generalization.

## Augmentation Techniques

Typical transformations include:

* Rotation
* Shift
* Scale
* Zoom
* Brightness adjustment
* Contrast adjustment
* Blur
* Gaussian noise
* Background variation
* Minor affine transformations
* Cropping variation
* Ink intensity variation

Example augmentation pipeline:

```python
import albumentations as A

train_transform = A.Compose([
    A.Rotate(limit=12, p=0.5),
    A.ShiftScaleRotate(
        shift_limit=0.05,
        scale_limit=0.10,
        rotate_limit=10,
        p=0.5
    ),
    A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.4
    ),
    A.GaussianBlur(
        blur_limit=(3, 5),
        p=0.2
    ),
    A.GaussNoise(
        p=0.2
    ),
    A.Resize(224, 224)
])
```

> Important: augmentation is applied only to the training dataset. Validation data remains clean and unchanged.

---

# Baseline Model Exploration

Before selecting Swin Transformer, multiple architectures were explored.

---

## 1. ResNet + BiLSTM + Attention

This baseline combines:

```text
Image
  ↓
ResNet CNN Backbone
  ↓
Feature Sequence
  ↓
BiLSTM
  ↓
Attention Layer
  ↓
Classifier
```

### Advantages

* Strong CNN feature extraction.
* BiLSTM helps model sequential feature relationships.
* Attention highlights informative regions.

### Limitations

* Images must be converted into a sequence representation.
* Spatial relationships may weaken after flattening.
* Kannada characters require rich two-dimensional structural understanding.
* More complex than necessary for isolated character recognition.

---

## 2. EfficientNet + CBAM

EfficientNet is an efficient CNN architecture using compound scaling.

CBAM stands for:

```text
Convolutional Block Attention Module
```

CBAM adds:

* Channel attention
* Spatial attention

Architecture:

```text
Image
  ↓
EfficientNet Backbone
  ↓
CBAM Attention Module
  ↓
Global Pooling
  ↓
Classification Head
```

### Advantages

* Efficient parameter usage.
* Better attention to important stroke regions.
* Improved CNN baseline performance.
* Good for local texture and shape features.

### Limitations

* CNN receptive fields are still primarily local.
* Complex ligatures require broader context.
* Long-range relationships between distant strokes are difficult to model.
* Less effective than Swin Transformer for hierarchical character structure.

---

# Why Swin Transformer

A standard Vision Transformer applies global attention across all image patches. This can be computationally expensive and may require large datasets.

Swin Transformer solves this by using:

* Window-based self-attention
* Shifted windows
* Hierarchical feature maps
* Patch merging
* Multi-scale learning

This makes it suitable for handwritten Kannada recognition.

Swin Transformer learns:

```text
Local Stroke Details
        +
Global Character Structure
        +
Hierarchical Multi-Scale Features
```

This is important because Kannada characters may contain:

* Small dots
* Curves
* Loops
* Vowel modifiers
* Connected components
* Ligatures
* Fine stroke differences

---

# Swin Transformer Architecture

```text
Input Image: 224 × 224 × 3
        │
        ▼
Patch Partition
        │
        ▼
Linear Embedding
        │
        ▼
Swin Transformer Stage 1
Window Multi-Head Self Attention
        │
        ▼
Patch Merging
        │
        ▼
Swin Transformer Stage 2
Shifted Window Attention
        │
        ▼
Patch Merging
        │
        ▼
Swin Transformer Stage 3
Hierarchical Feature Learning
        │
        ▼
Patch Merging
        │
        ▼
Swin Transformer Stage 4
High-Level Semantic Features
        │
        ▼
Global Average Pooling
        │
        ▼
512-Dimensional Embedding Head
        │
        ├───────────────► Softmax Classification Head
        │                       │
        │                       ▼
        │                  113 Kannada Classes
        │
        └───────────────► Centroid Refinement
                                │
                                ▼
                        Calibrated Top-3 Predictions
```

---

## Patch Partition

The input image is divided into small non-overlapping patches.

Instead of processing every pixel individually, the model processes patch embeddings.

```text
224 × 224 Image
        ↓
Patch Partition
        ↓
Patch Tokens
        ↓
Transformer Processing
```

---

## Window-Based Self-Attention

Instead of applying attention across the entire image, Swin Transformer applies attention within local windows.

```text
Image Feature Map
        ↓
Split into Windows
        ↓
Self-Attention inside each Window
```

This reduces computational cost.

---

## Shifted Window Attention

In alternate layers, the attention windows are shifted.

```text
Layer 1:
Normal Windows

Layer 2:
Shifted Windows
```

This allows information to flow between neighboring windows.

For Kannada characters, shifted windows help connect stroke patterns that may lie in different local regions.

---

## Patch Merging

Patch merging reduces spatial resolution and increases feature depth.

```text
High Resolution + Low Semantic Features
        ↓
Patch Merging
        ↓
Lower Resolution + Higher Semantic Features
```

This allows the model to learn both fine strokes and high-level character structure.

---

# Swin V1, V2, and V3

The project progresses through three Swin Transformer versions.

| Version | Main Features                                                    | Purpose                                      |
| ------- | ---------------------------------------------------------------- | -------------------------------------------- |
| Swin V1 | Swin Tiny baseline                                               | Establish Transformer baseline               |
| Swin V2 | Cosine scheduler + focal loss                                    | Improve convergence and hard-sample learning |
| Swin V3 | Embedding head + centroid refinement + calibration + diagnostics | Final reliable deployment model              |

---

## Swin V1

Swin V1 serves as the initial Transformer baseline.

### Features

* Swin Tiny backbone
* Standard classification head
* Hierarchical attention
* Patch merging
* Window-based attention
* Shifted-window attention

### Role

Swin V1 establishes that hierarchical Transformer models are better suited than CNN baselines for Kannada handwritten character recognition.

---

## Swin V2

Swin V2 improves the training strategy.

### Improvements

* Focal Loss
* Cosine annealing learning-rate scheduler
* Warmup strategy
* Mixed precision training
* Gradient clipping
* Better handling of difficult classes

### Why It Matters

Swin V2 improves learning stability and shifts attention toward hard-to-classify samples.

---

## Swin V3

Swin V3 is the final model.

### Improvements

* 512-dimensional embedding head
* Centroid-based refinement
* Temperature scaling
* Expected Calibration Error evaluation
* Hard-sample mining
* Robustness testing
* Enhanced diagnostics
* Deployment-ready checkpoint export

### Why It Was Selected

Swin V3 provides more than high accuracy.

It provides:

* Reliable confidence scores
* Strong robustness
* Stable Top-3 predictions
* Better separation of visually similar classes
* Improved deployment readiness
* Better support for feedback-driven retraining

---

# Why Focal Loss Instead of Cross-Entropy

## Cross-Entropy Loss

Cross-entropy treats all samples equally.

```text
Easy Sample → Contributes to Loss
Hard Sample → Contributes to Loss
```

In high-accuracy models, easy examples dominate the loss.

This can reduce the focus on difficult Kannada character pairs.

---

## Focal Loss

Focal Loss reduces the contribution of easy examples and focuses more on hard samples.

```text
Easy Correct Sample
        ↓
Lower Loss Contribution

Hard Misclassified Sample
        ↓
Higher Loss Contribution
```

Mathematically:

```text
FL(pt) = -α(1 - pt)^γ log(pt)
```

Where:

* `pt` is the predicted probability for the correct class
* `α` balances class importance
* `γ` controls how much focus is placed on difficult samples

---

## Why Focal Loss Helps Here

Focal Loss is especially useful because Kannada handwriting includes:

* Similar strokes across classes
* Rare compound characters
* Small modifier differences
* Difficult ligature-like forms
* Ambiguous handwritten shapes

Instead of allowing easy examples to dominate training, focal loss pushes the model to learn difficult boundaries.

---

# Centroid Refinement

The final Swin V3 model produces:

1. Softmax probabilities
2. A 512-dimensional feature embedding

For every class, a centroid is calculated in embedding space.

```text
Class Images
      ↓
Feature Embeddings
      ↓
Average Embedding
      ↓
Class Centroid
```

During inference:

```text
Input Image
      ↓
Swin Transformer
      ↓
Softmax Probability
      +
Embedding Similarity to Class Centroids
      ↓
Refined Prediction Score
      ↓
Final Top-3 Predictions
```

## Why Centroid Refinement Helps

Centroid refinement improves prediction consistency for visually similar Kannada characters.

It helps:

* Reduce overconfident mistakes
* Improve Top-3 stability
* Identify embedding-level confusion
* Improve hard-sample analysis
* Support future active learning

---

# Calibration and Expected Calibration Error

## Why Calibration Matters

A model can be accurate but poorly calibrated.

For example:

```text
Model Confidence = 99%
Actual Correctness = 80%
```

This means the model is overconfident.

For OCR deployment, confidence must be trustworthy.

---

## Temperature Scaling

Temperature scaling adjusts logits before softmax.

```text
Calibrated Probability = Softmax(Logits / Temperature)
```

The optimal temperature is learned using validation data.

For the final model:

```text
Temperature = 0.5
ECE ≈ 0.001
```

---

## Expected Calibration Error

Expected Calibration Error measures how closely confidence matches actual correctness.

```text
Lower ECE = Better Calibration
```

| ECE Value    | Interpretation                       |
| ------------ | ------------------------------------ |
| High ECE     | Model confidence is unreliable       |
| Moderate ECE | Confidence is somewhat useful        |
| Low ECE      | Confidence aligns well with accuracy |
| ECE ≈ 0.001  | Excellent calibration                |

The final Swin V3 model achieves:

```text
ECE ≈ 0.001
```

This means model confidence can be used more reliably for:

* Human verification
* Automated rejection of uncertain samples
* Feedback prioritization
* Active learning
* OCR workflow automation

---

# Robustness and Stress Testing

A real OCR system must work under more than clean benchmark images.

The final model was tested under several perturbations.

| Stress Condition       | Accuracy | Macro-F1 |
| ---------------------- | -------: | -------: |
| Ink Darkening          |   0.9823 |   0.9982 |
| Background Jitter      |   0.9985 |   0.9986 |
| Zoom / Crop Distortion |   0.9971 |   0.9972 |

---

## Stress Conditions

### Ink Darkening

Simulates:

* Dark pen strokes
* Marker-like writing
* Ink bleed
* Heavy pen pressure

### Background Jitter

Simulates:

* Uneven paper texture
* Scan artifacts
* Camera background noise
* Ruled paper variation

### Zoom and Crop Distortion

Simulates:

* Improper camera framing
* Partial cropping
* Scale variation
* Mobile capture errors

---

## Interpretation

The model remains robust across all tested perturbations.

The largest performance decrease occurs under ink darkening, which is expected because extreme stroke thickness can alter the internal shape of handwritten characters.

However, performance remains strong, demonstrating that the model learns stable visual representations.

---

# Metrics Comparison

> Replace the values below with exact values from your experiment notebook if needed.

| Model                       | Train Accuracy | Validation Accuracy | Validation Loss |  Macro-F1 |           ECE | Notes                                |
| --------------------------- | -------------: | ------------------: | --------------: | --------: | ------------: | ------------------------------------ |
| ResNet + BiLSTM + Attention |       Baseline |            Baseline |          Higher |     Lower | Not evaluated | Sequence-based CNN baseline          |
| EfficientNet + CBAM         |         Strong |              Strong |        Moderate |    Strong | Not evaluated | Better local attention               |
| Swin V1                     |         Higher |              Higher |           Lower |    Higher | Not evaluated | Hierarchical Transformer baseline    |
| Swin V2                     |       Improved |            Improved |           Lower |  Improved |     Evaluated | Focal loss + cosine schedule         |
| Swin V3                     |      Very High |              ~99.8% |          Lowest | Very High |        ~0.001 | Final model with centroid refinement |

---

# Evaluation Metrics

The project evaluates multiple metrics rather than relying only on accuracy.

## Accuracy

```text
Accuracy = Correct Predictions / Total Predictions
```

Measures overall classification correctness.

---

## Precision

```text
Precision = True Positives / (True Positives + False Positives)
```

Measures how many predicted labels are correct.

---

## Recall

```text
Recall = True Positives / (True Positives + False Negatives)
```

Measures how many actual class samples are detected correctly.

---

## F1 Score

```text
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

Useful for balanced class evaluation.

---

## Macro-F1

Macro-F1 calculates F1 independently for each class and then averages the results.

This is important because all Kannada classes should be evaluated equally.

---

## Top-3 Accuracy

Top-3 accuracy checks whether the correct class appears among the three highest predictions.

This is valuable for:

* Human verification
* Feedback interfaces
* Ambiguous handwriting
* OCR correction workflows

---

## Confusion Matrix

A normalized confusion matrix is used to identify:

* Frequently confused character pairs
* Weak classes
* Rare ligatures
* Stroke-overlap errors
* Vowel modifier confusion

---

# Gradio Deployment

The project includes a Gradio-based user interface.

Run the application:

```bash
python app.py
```

The application opens in the browser.

---

## Tab 1: Character Recognition

This tab supports single handwritten character recognition.

### Features

* Upload handwritten character image
* Preview input image
* Run preprocessing
* Display predicted character
* Display confidence score
* Display Top-3 predictions
* Submit feedback
* Select alternate prediction
* Enter manual correction
* Save feedback

Workflow:

```text
Upload Character Image
        ↓
Preprocessing
        ↓
Swin V3 Prediction
        ↓
Centroid Refinement
        ↓
Temperature Calibration
        ↓
Top-3 Predictions
        ↓
User Feedback
        ↓
TinyDB / CSV Storage
```

---

## Tab 2: Word Recognition and Progressive Segmentation

This tab supports prototype word-level OCR.

### Features

* Upload handwritten word image
* Generate segmentation candidates
* Test splits for:

```text
k = 2, 3, 4, 5
```

* Display cropped character segments
* Display Top-3 predictions for each segment
* Reconstruct candidate words
* Select best segmentation
* Enter manual corrected word
* Save segmentation feedback
* Export JSON feedback

Workflow:

```text
Upload Word Image
        ↓
Generate Candidate Splits
        ↓
Segment into Character Slices
        ↓
Run Character OCR on Each Slice
        ↓
Reconstruct Candidate Word
        ↓
User Selects Best Split
        ↓
Save Feedback
```

---

# Feedback-Driven Learning

The system stores user corrections for future retraining.

Feedback storage formats:

```text
segmentation_feedback.json
segmentation_feedback.csv
```

Character-level feedback may include:

```json
{
  "image_path": "sample.png",
  "predicted_class": "ka",
  "confidence": 0.97,
  "correct_label": "kha",
  "user_feedback": "manual_correction"
}
```

This feedback can later be used to:

* Identify weak classes
* Collect difficult samples
* Improve augmentation
* Retrain the model
* Expand the dataset
* Build active learning pipelines






# Requirements

Example `requirements.txt`:

```txt
torch
torchvision
timm
opencv-python
albumentations
numpy
pandas
scikit-learn
matplotlib
seaborn
gradio
tinydb
Pillow
tqdm
```

---

# Training

Train the final Swin V3 model:

```bash
python train.py \
  --model swin_v3 \
  --epochs 50 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --loss focal \
  --num_classes 113
```

Example training configuration:

```python
CONFIG = {
    "model": "swin_v3",
    "image_size": 224,
    "num_classes": 113,
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 1e-4,
    "optimizer": "AdamW",
    "loss": "FocalLoss",
    "scheduler": "CosineAnnealingLR",
    "mixed_precision": True,
    "gradient_clipping": True
}
```

---

# Inference

Run character prediction:

```bash
python evaluate.py \
  --checkpoint checkpoints/swin_v3.pth \
  --image sample.png
```

Example output:

```text
Top-1 Prediction: ಕ
Confidence: 0.998

Top-3 Predictions:
1. ಕ — 0.998
2. ಖ — 0.001
3. ಗ — 0.001
```

---

# Hardware Requirements

Recommended hardware:

| Component | Recommended                                 |
| --------- | ------------------------------------------- |
| CPU       | Intel i5/i7 or AMD Ryzen 5/7                |
| RAM       | 16 GB minimum                               |
| GPU       | NVIDIA RTX 3060 / Tesla T4 / V100 or better |
| Storage   | 50 GB+                                      |
| CUDA      | Recommended for training                    |
| OS        | Linux, Windows, Google Colab                |

Mixed precision training is recommended for faster training and lower GPU memory usage.

---

# Limitations

Current limitations include:

* The system focuses primarily on isolated character recognition.
* Word-level OCR uses prototype progressive segmentation rather than a learned end-to-end sequence model.
* No language model is used for contextual Kannada word correction.
* No full-page layout analysis.
* No line segmentation or paragraph recognition.
* Some rare compound characters remain difficult.
* Performance may reduce on extremely poor-quality scans.
* Dataset diversity can be expanded with more writers, devices, papers, and natural document images.
* Larger Swin models require more compute and memory.

---

# Future Work

Future improvements may include:

* End-to-end word recognition using Transformer decoders.
* Kannada language-model-based contextual correction.
* CTC-based OCR pipelines.
* Learned segmentation networks.
* Full-page OCR.
* Line and paragraph segmentation.
* Layout analysis.
* Self-supervised pretraining on unlabeled Kannada documents.
* Active learning from low-confidence feedback samples.
* Mobile deployment using quantization.
* Edge deployment using TensorRT or ONNX.
* Knowledge distillation for smaller models.
* Multilingual OCR for Telugu, Malayalam, Tamil, and other Indic scripts.
* Camera-based real-time OCR application.

---

# Research Takeaways

This project demonstrates that reliable OCR requires more than a high accuracy score.

A strong OCR system should include:

```text
High-Quality Dataset
        +
Balanced Classes
        +
Strong Architecture
        +
Hard-Sample Learning
        +
Calibration
        +
Robustness Testing
        +
Human Feedback
        +
Continuous Improvement
```

The final Swin V3 system combines:

* Hierarchical visual learning
* Local and global attention
* Focal loss
* Centroid refinement
* Temperature scaling
* ECE evaluation
* Stress testing
* Feedback-driven learning

This makes AksharaVision suitable as a foundation for future Kannada OCR research and deployment.

---

# Citation

```bibtex
@project{aksharavision2026,
  title={AksharaVision: Robust Kannada Handwritten Character Recognition using Swin Transformer},
  author={Shrihari M V and Lavanya N and Arati Balaji and Akash Suragi},
  year={2026},
  institution={Bangalore Institute of Technology},
  department={Department of CSE Data Science}
}
```

---

# Acknowledgements

* Bangalore Institute of Technology
* Department of CSE (Data Science)
* Project guide and faculty mentors
* Contributors who provided handwritten Kannada samples
* Open-source communities behind PyTorch, OpenCV, Albumentations, timm, and Gradio

---

# License

This project is intended for academic and research use.

Before redistributing the dataset, ensure that contributor consent, handwriting privacy, and institutional data policies are respected.

```
```
