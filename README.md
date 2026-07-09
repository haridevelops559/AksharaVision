
# AksharaVision  



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
## Key Contributions

- Curated a balanced Kannada handwritten-character dataset spanning **113 classes**; built an OpenCV contour-based sheet-to-character extraction pipeline and standardized the dataset to **500 images per class (56,500 images)**.

- Benchmarked **ResNet + BiLSTM + Attention**, **EfficientNet + CBAM**, and three **Swin Transformer** iterations; selected Swin V3 after correcting label/split issues, introducing class-stratified validation, and improving difficult-class learning.

- Replaced cross-entropy with **Focal Loss** and added a **512-dimensional embedding head**; incorporated centroid-neighbor analysis, Top-k evaluation, confusion matrices, confusing-pair reports, per-class metrics, and hard-sample review for error diagnosis.

- Calibrated V3 probabilities using **temperature scaling** and evaluated reliability with **ECE** and multiclass Brier score; stress-tested robustness under ink, background, and zoom/crop perturbations, and added occlusion-sensitivity maps for prediction-level visual interpretability.

- Built an earlier two-tab Gradio prototype for **single-character OCR** and **word-level OCR via progressive segmentation**; supported candidate splits, segment-level Top-3 predictions, reconstructed word hypotheses, and JSON/CSV feedback capture.

- Extended the interface into a three-tab Gradio workflow for **Character OCR**, **Explain Prediction**, and **Deployment Monitoring**; added calibrated Top-3 predictions, input-quality checks, Accept/Review/Retake routing, occlusion explanations, and local human-feedback capture.

- Designed privacy-aware anonymous inference telemetry for **latency, calibrated confidence, Top-1/Top-2 margin, decision route, blur, brightness, ink ratio, aspect ratio, and predicted label**; user-uploaded images are not stored by default.

- Added a validation-distribution baseline and **Population Stability Index (PSI)** drift checks for confidence, uncertainty margin, blur, brightness, and ink ratio to flag when live inputs differ from held-out evaluation data.

- Tracked the final **Swin V3 release** with :contentReference[oaicite:0]{index=0}, recording classification, calibration, robustness, deployment latency, throughput, model-footprint, and diagnostic artifacts for reproducible release validation.

---
## System Architecture and End-to-End Pipeline

AksharaVision is designed as an end-to-end Kannada handwritten OCR system that covers dataset construction, model development, reliability validation, explainability, interactive inference, and post-deployment monitoring.

```text
Handwritten Kannada Sheets / Character Images
        ↓
PDF and Image Ingestion
        ↓
OpenCV Contour Detection, Sheet Segmentation, and Character Cropping
        ↓
Dataset Cleaning, Label Verification, and Class Balancing
        ↓
113-Class Dataset (500 Images per Class; 56,500 Images)
        ↓
Train / Validation Split with Class-Stratified Representation
        ↓
Preprocessing and Augmentation
Resize • Normalize • Rotation • Contrast • Background Variation • Crop Robustness
        ↓
Architecture Benchmarking
ResNet + BiLSTM + Attention
        ↓
EfficientNet + CBAM
        ↓
Swin Transformer V1 → V2 → V3
        ↓
Final Swin V3 Training
Focal Loss • Regularization • Embedding Head • Fine-Tuning
        ↓
512-Dimensional Embedding Space
        ↓
Centroid-Neighbor Analysis and Similar-Class Context
        ↓
Held-Out Evaluation and Error Diagnostics
Top-1 / Top-2 / Top-3 • Macro-F1 • Per-Class Metrics
Confusion Matrix • Confusing Pairs • Hard Samples
        ↓
Reliability Validation
Temperature Scaling • ECE • Multiclass Brier Score
        ↓
Robustness and Interpretability
Ink / Background / Zoom Stress Tests • Occlusion Sensitivity
        ↓
W&B Swin V3 Release Validation
Metrics • Calibration • Robustness • Latency • Throughput • Model Footprint
        ↓
Gradio Application Layer
        ├── Tab 1: Character OCR
        ├── Tab 2: Word OCR and Progressive Segmentation
        ├── Tab 3: Explain Prediction and Deployment Monitoring
        ↓
Inference Decision Support
Calibrated Top-3 • Confidence Margin • Input Quality Checks
Accept • Review Required • Retake Image
        ↓
Privacy-Aware Telemetry and Feedback
Latency • Confidence • Margin • Blur • Brightness • Ink Ratio
Feedback CSV / JSON / TinyDB
        ↓
Drift Monitoring and Future Retraining
Validation Baseline • PSI Drift Checks • Curated Hard Samples
---
```
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
## Gradio Interface Evolution

AksharaVision evolved from a focused two-tab inference demo into a three-tab, reliability-aware OCR interface that separates prediction, explanation, and operational monitoring.

## Earlier Two-Tab Gradio Interface

The earlier Gradio application focused on extending character-level Kannada OCR into a prototype word-recognition workflow with progressive segmentation and feedback collection.

| Tab / Component | Purpose | User Inputs | System Workflow | Outputs | Feedback / Storage |
|---|---|---|---|---|---|
| **Tab 1: Character Recognition** | Recognize one handwritten Kannada character at a time | Character image upload; optional alternate prediction selection or manual correction | Image preprocessing → Swin V3 inference → centroid refinement → temperature calibration → Top-3 ranking | Input preview, predicted character, confidence score, and Top-3 predictions | Users can confirm an alternate Top-k prediction or enter a manual correction; feedback is stored in TinyDB and CSV for later review |
| **Image preprocessing** | Standardize input before inference | Uploaded character image | Resize, normalization, and image preparation for the Swin V3 model | Model-ready image tensor | No direct user feedback; preprocessing issues can be inferred from incorrect or low-confidence cases |
| **Swin V3 prediction** | Produce class probabilities across the character label space | Preprocessed image tensor | Swin Transformer extracts visual features and predicts class logits | Predicted class and raw class scores | Incorrect predictions can be captured through the character feedback flow |
| **Centroid refinement** | Add embedding-space similarity context | Model embedding and predicted class | Compares learned representation with class-centroid neighbors | Similar-class context to support ambiguous-character review | Helps identify visually related classes for later confusion analysis |
| **Temperature calibration** | Improve confidence interpretability | Model logits | Applies learned temperature scaling before probability ranking | Calibrated confidence and calibrated Top-3 predictions | Supports more meaningful confidence review before users submit feedback |
| **Tab 2: Word Recognition and Progressive Segmentation** | Prototype word-level OCR by splitting a handwritten word into character candidates | Word image upload; preferred segmentation selection; optional corrected word | Candidate split generation → character slicing → character OCR for each slice → candidate-word reconstruction | Segment previews, Top-3 predictions per segment, reconstructed word candidates, and selected split | Users can select the best split, enter a corrected word, and export segmentation feedback |
| **Candidate split generation** | Explore multiple possible character boundaries in a word image | Uploaded word image | Generates candidate segmentations for `k = 2, 3, 4, 5` character slices | Multiple segmentation hypotheses | User selection identifies which segmentation is most plausible |
| **Segment-level OCR** | Recognize each candidate character crop | Character slices from each candidate split | Runs the character-recognition pipeline independently on every crop | Top-3 predictions for each segment | Errors reveal whether failures originate from segmentation or character classification |
| **Candidate-word reconstruction** | Convert segment predictions into word hypotheses | Segment-level Top-3 predictions | Combines leading character predictions for each segmentation candidate | Reconstructed candidate words | User can choose the best candidate or provide a manual corrected word |
| **Character feedback loop** | Capture label corrections for single-character OCR | Top-1/Top-2/Top-3 selection or manual class label | Saves prediction and correction metadata | Reviewable character-level feedback records | TinyDB / CSV storage for future error analysis and curated retraining |
| **Segmentation feedback loop** | Capture word-level split and transcription corrections | Selected split and optional corrected word | Saves segmentation choice, predicted candidates, and correction | JSON and CSV feedback exports | `segmentation_feedback.json` and `segmentation_feedback.csv` |
| **Future learning use** | Turn reviewed errors into improvement candidates | Stored character and segmentation feedback | Analyze weak classes, hard handwriting, and segmentation failures | Curated data for targeted augmentation, retraining, and active-learning experiments | Feedback is collected for later review; it should not automatically retrain or modify the deployed model |




### Updated Three-Tab Workflow

| Tab | User Action | System Behavior | Output |
|---|---|---|---|
| **Character OCR** | Upload a handwritten Kannada character | Runs calibrated Swin V3 inference, quality checks, Top-3 ranking, and uncertainty routing | Prediction, alternatives, confidence, decision status, quality signals, and similar-class hints |
| **Explain Prediction** | Upload an image and request explanation | Masks image patches and measures confidence drop for the predicted class | Occlusion-sensitivity heatmap and interpretation summary |
| **Deployment Monitoring** | Refresh dashboard after inference requests | Aggregates local telemetry, calculates runtime statistics, evaluates PSI drift, and logs a sanitized W&B summary | Request count, latency statistics, review rate, quality trends, drift indicators, and telemetry export |

### Design Principle

The updated interface does not claim that confidence or explainability guarantees correctness. Instead, it combines calibrated confidence, Top-k alternatives, image-quality checks, occlusion sensitivity, human review, and telemetry monitoring to make OCR predictions more inspectable and operationally responsible.



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
# Swin V1, V2, and V3

The project progresses through three Swin Transformer versions.

| Version | Main Features                                                    | Purpose                                      |
| ------- | ---------------------------------------------------------------- | -------------------------------------------- |
| Swin V1 | Swin Tiny baseline                                               | Establish Transformer baseline               |
| Swin V2 | Cosine scheduler + focal loss                                    | Improve convergence and hard-sample learning |
| Swin V3 | Embedding head + centroid refinement + calibration + diagnostics | Final reliable deployment model              |

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

## V3 Reliability, Explainability, and Deployment Updates

AksharaVision V3 extends the core Swin Transformer OCR pipeline with release validation, reliability analysis, interpretability diagnostics, and lightweight deployment monitoring.

### Release Validation and Experiment Tracking

The final Swin V3 release was tracked with Weights & Biases (W&B) to version evaluation metrics, deployment measurements, and diagnostic artifacts.

| Metric | V3 Release Result |
|---|---:|
| Top-1 Accuracy | 99.89% |
| Top-2 Accuracy | 99.93% |
| Top-3 Accuracy | 99.95% |
| Macro-F1 | 0.9989 |
| Expected Calibration Error (ECE, 15 bins) | 0.00065 |
| Multiclass Brier Score | 0.00112 |
| Single-image Latency | 36.11 ms |
| Batch Throughput | 242.09 images/sec |
| Model Parameters | 27.97M |
| Checkpoint Size | 106.78 MB |

---
### Saved Occlusion Sensitivity  Artifacts

| Artifact | Purpose |
|---|---|
| `evaluation_prediction_explanation_manifest.csv` | Prediction-level audit trail containing true label, predicted label, confidence, correctness, and explanation paths. |
| `per_class_occlusion_summary.csv` | Class-level overview across all 113 labels, including evaluation count, accuracy, representative confidence, and hard-example confidence. |
| `representative_correct/` | Occlusion visualizations for representative correct predictions. |
| `hard_examples/` | Occlusion visualizations for low-confidence or misclassified predictions. |

> **Interpretation note:** Occlusion sensitivity measures how much masking a region changes the model's confidence for its original prediction. It is useful for debugging and qualitative review, but it should not be presented as a causal explanation of model reasoning.

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


## End-to-End Technical Overview

| Area | Implementation / Technique | V3 Upgrade or Nuance | Proof of Work / Artifact | Applied ML Signal |
|---|---|---|---|---|
| **Dataset curation** | OpenCV contour-based sheet segmentation, character cropping, cleaning, label verification, and balancing | Built a 113-class dataset with 500 images per class (56,500 total); preserved class mapping through JSON-aligned processing | Dataset manifests, class JSON, extraction pipeline | Data-centric ML, dataset integrity, reproducibility |
| **Validation design** | Class-stratified train/validation split | V3 uses a proper held-out split with representation from all 113 classes; addresses earlier split/label-quality limitations | Split manifests and per-class support counts | Evaluation discipline; avoids relying only on training accuracy |
| **Architecture experimentation** | ResNet + BiLSTM + Attention, EfficientNet + CBAM, Swin Transformer iterations | Compared CNN/sequence-attention and transformer approaches before selecting Swin V3 | Model comparison notes, notebooks, checkpoints | Architecture selection based on evidence rather than one-model training |
| **Swin Transformer V3** | `swin_tiny_patch4_window7_224` backbone with global pooling | Shifted-window self-attention captures local stroke structure while enabling cross-window context; suitable for visually similar handwritten glyphs | Swin V3 checkpoint and model definition | Vision Transformer understanding, transfer learning, fine-tuning |
| **Classification head** | 512-dimensional embedding layer, BatchNorm, ReLU, dropout, linear classifier | Separates representation learning from classification and enables embedding-space diagnostics | Embeddings, centroid-neighbor CSV | Representation learning and diagnostic reasoning |
| **Loss-function upgrade** | Focal Loss instead of standard cross-entropy | V3 emphasizes difficult or visually similar samples rather than allowing easy examples to dominate optimization | V3 training configuration and release notes | Imbalanced/hard-example learning |
| **Prediction refinement** | Embedding centroid-neighbor analysis and calibrated Top-k ranking | Similar-class context supports review of ambiguous glyphs; it is diagnostic context, not a replacement for classifier probabilities | `centroid_top3_neighbors.csv` | Metric-space analysis, uncertainty-aware UX |
| **Core evaluation** | Top-1/Top-2/Top-3 accuracy, macro-F1, per-class precision, recall, F1 | Evaluates both overall and class-balanced performance across all 113 labels | `per_class_detailed_metrics.csv`, W&B release metrics | Multiclass evaluation beyond accuracy |
| **Error diagnostics** | Normalized confusion matrix, ranked confusing pairs, hard-sample analysis | Identifies recurring true-class → predicted-class errors and separates low-confidence from confidently wrong cases | `confusion_matrix_113.png`, `confusing_class_pairs.csv`, hard-example CSV/preview | Systematic error analysis and data-improvement planning |
| **Reliability** | Temperature scaling, Expected Calibration Error, multiclass Brier score | Converts raw logits into more meaningful confidence estimates before displaying confidence or routing decisions | Calibration checkpoint, ECE/Brier metrics | Probabilistic ML and calibration awareness |
| **Robustness testing** | Ink darkening, background brightness, zoom/crop perturbations | Tests whether performance remains stable under plausible handwriting capture variation | `robustness_stress_metrics.csv` | Distribution-shift and robustness evaluation |
| **Interpretability** | Occlusion sensitivity for individual Swin V3 predictions | Masks image patches and measures predicted-class confidence drop; useful for checking whether evidence aligns with meaningful strokes | Occlusion manifests, representative-correct and hard-example heatmaps | Post-hoc visual interpretability with appropriate caveats |
| **Earlier Gradio: Tab 1** | Single-character OCR | Image upload, preprocessing, calibrated Top-3 predictions, alternate selection, manual correction | Character feedback CSV / TinyDB | Interactive inference and human-in-the-loop feedback |
| **Earlier Gradio: Tab 2** | Prototype word OCR with progressive segmentation | Generates candidate splits (`k=2–5`), runs character OCR per segment, reconstructs word candidates, stores corrections | Segmentation feedback JSON / CSV | OCR pipeline thinking beyond isolated classification |
| **Current Gradio: Character OCR** | Calibrated Top-3 inference with quality gates | Uses confidence, Top-1/Top-2 margin, blur, brightness, and ink ratio to route predictions as **Accept**, **Review Required**, or **Retake Image** | Local telemetry and UI decision outputs | Responsible inference design |
| **Current Gradio: Explain Prediction** | On-demand occlusion visualization | Connects an individual prediction with confidence-sensitive image regions and similar-class context | Occlusion plot and explanation summary | Explainable AI integrated into the product layer |
| **Current Gradio: Deployment Monitoring** | Runtime dashboard for aggregate inference behavior | Summarizes latency, confidence, review rate, input-quality trends, and drift signals | Telemetry CSV, monitoring table | MLOps observability and deployment awareness |
| **Inference telemetry** | Anonymous logging of latency, confidence, margin, decision route, blur, brightness, ink ratio, aspect ratio, and predicted label | Avoids storing uploaded images by default; separates operational metadata from raw user content | `inference_telemetry.csv` | Privacy-aware monitoring design |
| **Drift monitoring** | Held-out validation baseline with Population Stability Index checks | Compares live confidence, margin, blur, brightness, and ink-ratio distributions against evaluation data | `deployment_drift_baseline.json`, PSI table | Post-deployment distribution-shift detection |
| **Experiment tracking** | Weights & Biases V3 release validation | Records model quality, calibration, robustness, latency, throughput, parameters, checkpoint size, and diagnostics as a release record | W&B V3 release run and exported metrics | Reproducibility, experiment tracking, model release discipline |
| **Feedback-to-retraining loop** | Local CSV, JSON, and TinyDB correction capture | Feedback is curated for review, not automatically used to update the deployed model; supports future active-learning-style data collection | Character and segmentation feedback files | Safe iterative improvement and data flywheel design |

### What V3 Demonstrates

| Capability | Evidence |
|---|---|
| **Model development** | CNN and transformer benchmarking, Swin V3 fine-tuning, Focal Loss, embedding head |
| **Evaluation maturity** | Stratified validation, Top-k metrics, macro-F1, per-class analysis |
| **Failure analysis** | Confusion matrix, confusing pairs, hard samples, centroid neighbors |
| **Reliability awareness** | Temperature scaling, ECE, Brier score, uncertainty routing |
| **Vision-specific validation** | Robustness tests for handwriting and image-capture variation |
| **Interpretability** | Occlusion sensitivity tied to individual predictions and error review |
| **Product implementation** | Character OCR, word-segmentation prototype, feedback workflows |
| **Deployment thinking** | Latency profiling, quality gates, telemetry, drift baseline, PSI monitoring |
| **MLOps practice** | W&B release tracking, versioned artifacts, lightweight repository hygiene |









---


## Interpretation

The model remains robust across all tested perturbations.

The largest performance decrease occurs under ink darkening, which is expected because extreme stroke thickness can alter the internal shape of handwritten characters.

However, performance remains strong, demonstrating that the model learns stable visual representations.


### Reliability and Decision Support

- Applied **temperature scaling** to calibrate confidence scores before inference.
- Evaluated reliability using **ECE** and **multiclass Brier score**, rather than reporting accuracy alone.
- Added calibrated **Top-3 predictions** to support review of visually similar Kannada characters.
- Designed an uncertainty-aware routing policy:
  - **Accept** for confident, clear predictions
  - **Review Required** for low confidence or a small Top-1 / Top-2 margin
  - **Retake Image** for low-quality inputs such as blur, insufficient ink, or overly dark crops

### Explainability and Error Analysis

- Added **occlusion sensitivity** explanations for Swin V3 predictions; masked image patches are evaluated to identify regions that most influence predicted-class confidence.
- Retained per-class precision, recall, F1-score, normalized confusion matrices, confusing-class-pair analysis, and hard-example review.
- Used embedding-centroid nearest-neighbor analysis to identify visually similar character classes and support Top-k review.
- Evaluated robustness under ink contrast, background brightness, and zoom/crop perturbations.

### Occlusion Sensitivity Overview

Occlusion sensitivity is a post-hoc interpretability diagnostic for individual Swin V3 predictions. It masks small image patches one at a time and measures the reduction in calibrated confidence for the originally predicted class.

| Output / Field | Meaning | How to Interpret It | Recommended Action |
|---|---|---|---|
| Input image | Original handwritten Kannada character submitted for inference | Provides the visual context for the explanation | Inspect stroke quality, crop, background, and character completeness. |
| True label | Ground-truth class from the evaluation manifest | Available for held-out evaluation images only | Compare with predicted label to determine whether the explanation corresponds to a correct prediction. |
| Predicted label | Highest-probability class predicted by Swin V3 | The class whose confidence is evaluated during patch masking | Use alongside Top-2 and Top-3 alternatives for visually similar characters. |
| Calibrated confidence | Temperature-scaled probability of the predicted class | Higher confidence indicates stronger model certainty on the evaluation distribution | Low confidence should trigger review, especially if image quality is weak. |
| Correctness | Whether predicted label matches the true label | Separates explanations for correct predictions from explanations for failures | Review incorrect predictions and compare their highlighted regions with confusing classes. |
| Occlusion heatmap | Spatial map of confidence decrease after each patch is masked | Brighter regions indicate patches that caused a larger drop in predicted-class confidence | Check whether highlighted regions overlap meaningful strokes rather than borders, blank space, or noise. |
| Confidence drop | Difference between original confidence and confidence after masking a patch | Larger positive values indicate stronger sensitivity to that region | Use as relative evidence of importance; do not interpret it as causal proof. |
| Representative correct example | High-confidence correct sample selected for each class | Shows typical visual evidence used for a correctly classified character | Compare representative patterns across the 113 classes. |
| Hard example | Misclassified or low-confidence sample selected for a class | Highlights uncertain, ambiguous, or visually difficult handwriting | Use for targeted data review, augmentation ideas, and human-review UI design. |
| Per-class occlusion summary | One-row summary for each of the 113 classes | Combines evaluation sample count, class accuracy, representative confidence, and hard-example confidence | Identify classes that are accurate but fragile, or classes that need deeper error analysis. |
| Evaluation explanation manifest | Prediction-level record for evaluated samples | Links file path, true label, prediction, confidence, correctness, and saved explanation outputs | Makes interpretability analysis reproducible and auditable. |
| Similar-class context | Confusion pairs and centroid-neighbor diagnostics | Identifies visually nearby labels that may compete with the predicted class | Inspect whether errors align with known confusing character pairs. |

### Reading the Heatmap

| Heatmap Pattern | Plausible Interpretation | Caution |
|---|---|---|
| Bright regions align with key handwritten strokes | The prediction is sensitive to character-defining visual structure | This is supportive evidence, not a proof that the model learned linguistic rules. |
| Bright regions appear mainly on image borders | The model may be influenced by crop position, padding, or background artifacts | Review preprocessing consistency and augmentation coverage. |
| Bright regions focus on noise, shadows, or background | The model may rely on spurious visual cues | Inspect the sample, input-quality checks, and train/evaluation distribution. |
| Diffuse heatmap across most of the image | The model may use broad global shape information or lack localized evidence | Compare with confidence, Top-1/Top-2 margin, and similar-class alternatives. |
| Weak or nearly uniform heatmap | No single patch strongly changes confidence | This can occur for highly confident predictions or when the chosen patch size is too coarse. |
| Incorrect prediction with concentrated bright strokes | The model relied on meaningful strokes but mapped them to a visually similar wrong class | Review confusing pairs, Top-k predictions, and add targeted examples if the pattern recurs. |
| Incorrect prediction with background-focused heatmap | The model may be responding to nuisance cues rather than character structure | Flag for data-quality review and potential augmentation or crop normalization. |



### Deployment Monitoring

The Gradio deployment workflow is being extended with lightweight, privacy-aware monitoring.

- Logs anonymous inference metadata: model version, latency, calibrated confidence, Top-1 / Top-2 margin, decision route, blur score, brightness, ink ratio, and aspect ratio.
- Does **not** store uploaded user images by default.
- Supports human feedback for Top-1, Top-2, Top-3, or manually corrected labels.
- Links feedback to a prediction request ID for future error review and curated retraining.
- Defines a drift baseline from held-out evaluation data and compares live input distributions using Population Stability Index (PSI) for confidence, uncertainty margin, blur, brightness, and ink ratio.



---

 |



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


