# Camouflage Breaker — Project Journey

This document records the development path of Camouflage Breaker from the first segmentation approach to the final full-stack application.

## 1. Problem Definition

Camouflage Breaker was designed to detect objects that visually blend into their surroundings, localize the object, visualize the detected region, extract it, and identify the object class.

## 2. Dataset — COD10K-v3

The project uses COD10K-v3.

- 6,000 training images
- 4,000 test images
- Corresponding object masks
- 69-class object taxonomy used for classification

The dataset was verified before model development. The local verification found matching image/mask counts, no missing masks, no corrupted images or masks, and no image-mask dimension mismatches.

## 3. Initial Segmentation Approach — ResUNet

The first segmentation approach used a ResNet50 encoder with a U-Net decoder.

This stage established the basic segmentation workflow:

```text
Image → ResUNet → Probability Map → Binary Mask
```

The implementation and associated evaluation/test scripts are preserved in:

```text
historical/initial-resunet/
```

ResUNet is not used by the final production pipeline.

## 4. Improved Segmentation Approach — SINet-V2

The project then moved to the official SINet-V2 architecture for camouflaged object segmentation.

The final implementation uses the official SINet-V2 network with its Res2Net-based architecture and the project's fine-tuned checkpoint.

The active source is:

```text
models/sinetv2/source/
```

## 5. SINet-V2 Fine-Tuning

The final training run used:

- COD10K training set
- 40 epochs
- T4 GPU in Google Colab
- 90/10 validation split
- Seed 42
- Structure loss
- Cosine learning-rate scheduling

The best checkpoint was produced during the 40-epoch training run.

The final training script is:

```text
training/train_sinetv2_40.py
```

## 6. Segmentation Evaluation

The final SINet-V2 model was evaluated on all 4,000 COD10K test images.

| Metric | Result |
|---|---:|
| Test images | 4,000 |
| Mean IoU | 0.7191 |
| Mean Dice | 0.7657 |
| IoU ≥ 0.50 | 3,070 / 4,000 |
| IoU ≥ 0.70 | 2,619 / 4,000 |

These are segmentation metrics, not end-to-end classification accuracy.

The final evaluation utility is:

```text
training/evaluate_sinetv2_full.py
```

## 7. Object Extraction

The predicted segmentation mask is used to identify the object region.

The pipeline generates:

- binary mask
- object boundary
- visual overlay
- object crop

The crop becomes the input for classification.

## 8. Object Gate

A segmentation-only system can produce weak regions on images where no meaningful object is present.

To reduce false classifications, the project introduced an object-presence gate using:

- segmented area ratio
- largest connected-component ratio
- mean probability
- maximum probability

Only when the segmentation evidence passes the gate does the pipeline proceed with object classification.

## 9. Classification — ResNet50

The detected crop is passed to the trained ResNet50 classifier.

```text
Object Crop → ResNet50 → Class Probability → Class Name + Confidence
```

The final classifier supports 69 classes.

The class mapping is stored in:

```text
saved_models/class_mapping.json
```

The final classifier training script is:

```text
training/train_classifier_final.py
```

## 10. Complete AI Inference Pipeline

The final Python pipeline combines segmentation, gating, visualization, cropping, and classification.

```text
Input Image
    ↓
SINet-V2
    ↓
Segmentation Mask
    ↓
Object Gate
    ↓
Boundary / Overlay / Crop
    ↓
ResNet50
    ↓
Class + Confidence
```

Production inference code:

```text
inference/pipeline.py
```

## 11. FastAPI AI Service

The Python inference pipeline was exposed through FastAPI.

```text
POST /predict
GET  /health
```

The AI service runs on port 8000 during local development.

## 12. Spring Boot Backend

A Java 17 Spring Boot backend was added as the application/API layer.

The backend:

1. receives the uploaded image,
2. forwards it to FastAPI,
3. receives the prediction JSON,
4. returns the result to the frontend.

The backend runs on port 8080.

## 13. Frontend

A dedicated HTML/CSS/JavaScript frontend was built around the backend API.

The interface provides:

- drag-and-drop upload
- image preview
- detection processing state
- prediction name
- confidence
- segmentation visualization
- object crop
- downloadable result package
- new-scan workflow

## 14. End-to-End Application

The final architecture is:

```text
Browser
   ↓
Spring Boot :8080
   ↓
FastAPI :8000
   ↓
SINet-V2
   ↓
Object Gate
   ↓
ResNet50
   ↓
JSON + Result Images
   ↓
Spring Boot
   ↓
Browser
```

## 15. Why Historical Files Are Kept

Earlier approaches are intentionally preserved instead of deleted.

This makes the repository show the actual engineering progression:

```text
Initial approach
      ↓
Evaluation
      ↓
Model experimentation
      ↓
Final model selection
      ↓
Integrated inference
      ↓
Backend
      ↓
Frontend
      ↓
Complete application
```

Historical code is separated from production code so it remains useful without confusing the final runtime architecture.
