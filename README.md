# Camouflage Breaker

> **AI-powered camouflaged object detection, segmentation, and classification system**

Camouflage Breaker is a full-stack computer vision application designed to detect objects that visually blend into their surroundings. The system combines a deep segmentation model with a trained image classifier and exposes the complete inference pipeline through a modern web application.

The system takes an uploaded image, determines whether a meaningful camouflaged object is present, localizes it at pixel level, generates visual explanations, extracts the detected object, and predicts its class with a confidence score.

## Demo Flow

```text
Image Upload
     │
     ▼
Spring Boot Web Backend
     │
     ▼
FastAPI AI Service
     │
     ▼
SINet-V2 Segmentation
     │
     ├── Binary Mask
     ├── Object Gate
     ├── Boundary
     └── Orange/Red Overlay
     │
     ▼
Object Crop
     │
     ▼
ResNet50 Classification
     │
     ▼
Class + Confidence
     │
     ▼
Web Result Dashboard
```

## Project Development Journey

The repository intentionally separates the final production implementation from historical development work. Earlier approaches are preserved instead of deleted so the project documents its progression from the initial segmentation model to the final full-stack application.

The chronological development record is available in [docs/PROJECT_JOURNEY.md](docs/PROJECT_JOURNEY.md).

```text
Problem Definition
      ↓
COD10K-v3 Dataset
      ↓
Initial ResUNet Segmentation
      ↓
SINet-V2 Experiments
      ↓
40-Epoch SINet-V2 Fine-Tuning
      ↓
Segmentation Evaluation
      ↓
Object Extraction + Object Gate
      ↓
69-Class ResNet50
      ↓
Python Inference Pipeline
      ↓
FastAPI AI Service
      ↓
Spring Boot Backend
      ↓
Frontend
      ↓
End-to-End Application
```

Historical development code is stored under [historical/](historical/) and is not used by the production inference path.

## Key Features

- **Camouflaged object segmentation** using a fine-tuned SINet-V2 model.
- **Object presence gate** to prevent classification when the segmentation result does not meet the detection criteria.
- **Pixel-level mask generation** for the detected object.
- **Boundary visualization** around the detected region.
- **Orange/red explainability overlay** for visual interpretation.
- **Automatic object cropping** from the predicted segmentation mask.
- **69-class ResNet50 classifier** for object recognition.
- **Confidence score** for the predicted class.
- **Modern web interface** with drag-and-drop image upload, processing states, result cards, and downloadable analysis output.
- **Java/Python service separation** so the web backend and deep-learning inference service remain independently maintainable.
- **JSON API** for integrating the AI pipeline with other clients.

## Model Architecture

### 1. SINet-V2 — WHERE?

SINet-V2 performs camouflaged object segmentation and produces the spatial region containing the object.

The project fine-tunes the official SINet-V2 architecture on COD10K and uses the final prediction output from the network for inference.

**Input:** RGB image  
**Output:** Pixel-level object probability map → binary mask

### 2. Object Gate — IS THERE A VALID OBJECT?

The segmentation output is passed through a lightweight decision gate using:

- segmented area ratio
- largest connected-component ratio
- mean probability over the predicted region
- maximum probability

The gate prevents the classifier from returning an animal/object name when the segmentation evidence is too weak.

### 3. ResNet50 — WHAT?

The detected region is cropped and passed to a trained ResNet50 classifier.

**Input:** Extracted object crop  
**Output:** One of 69 trained classes + confidence

The classification mapping is stored in `saved_models/class_mapping.json`.

## Dataset

The project uses **COD10K-v3**, a large-scale camouflaged object detection dataset.

Required local structure:

```text
dataset/
├── Train/
│   ├── Image/
│   ├── GT_Object/
│   └── JSON/
└── Test/
    ├── Image/
    ├── GT_Object/
    └── JSON/
```

Dataset verification completed during development:

| Split | Images | Masks | Missing Masks | Corrupted Images | Dimension Mismatch |
|---|---:|---:|---:|---:|---:|
| Train | 6,000 | 6,000 | 0 | 0 | 0 |
| Test | 4,000 | 4,000 | 0 | 0 | 0 |

The dataset is intentionally not included in this repository.

## Segmentation Evaluation

The fine-tuned SINet-V2 model was evaluated on the complete 4,000-image COD10K test set.

| Metric | Result |
|---|---:|
| Test images evaluated | 4,000 |
| Missing masks | 0 |
| Mean IoU | **0.7191** |
| Mean Dice | **0.7657** |
| Minimum IoU | 0.0000 |
| Maximum IoU | 1.0000 |
| IoU ≥ 0.50 | 3,070 / 4,000 |
| IoU ≥ 0.70 | 2,619 / 4,000 |

These values describe segmentation performance and should not be interpreted as end-to-end classification accuracy.

## Project Structure

```text
Camouflage_Breaker/
│
├── ai_service/
│   └── main.py                  # FastAPI AI inference service
│
├── backend/
│   ├── src/main/java/
│   │   └── Camouflage/Breaker/Backend/
│   │       ├── BackendApplication.java
│   │       └── controller/
│   │           └── PredictionController.java
│   ├── src/main/resources/
│   │   └── static/              # Web interface served by Spring Boot
│   ├── pom.xml
│   ├── mvnw
│   └── mvnw.cmd
│
├── frontend/
│   ├── index.html
│   ├── css/style.css
│   └── js/app.js
│
├── inference/
│   ├── pipeline.py              # Final inference pipeline
│   └── test_pipeline.py
│
├── models/
│   ├── classifier.py
│   └── sinetv2/
│       └── source/              # SINet-V2 source used by the project
│
├── saved_models/
│   └── class_mapping.json
│
├── training/
│   ├── train_sinetv2_40.py       # Final SINet-V2 training
│   ├── evaluate_sinetv2_full.py  # Final 4,000-image evaluation
│   ├── train_classifier_final.py # Final classifier training
│   └── evaluate_pipeline.py      # Pipeline evaluation utilities
│
├── historical/
│   ├── initial-resunet/          # Initial segmentation approach
│   ├── segmentation-experiments/ # Earlier SINet-V2 / SAM 2 experiments
│   ├── classifier-development/   # Earlier classifier development
│   └── README.md
│
├── docs/
│   └── PROJECT_JOURNEY.md        # Chronological development story
│
├── utils/
│   ├── object_gate.py
│   └── check_json.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Technology Stack

### AI / Computer Vision

- Python
- PyTorch
- Torchvision
- OpenCV
- NumPy
- Pillow
- Albumentations
- Segmentation Models PyTorch
- scikit-learn
- tqdm

### Segmentation

- SINet-V2
- Res2Net backbone
- COD10K-v3

### Classification

- ResNet50
- 69-class object taxonomy
- Transfer-learning based architecture

### Backend

- Java 17
- Spring Boot
- Maven
- REST API
- Multipart image upload

### AI API

- FastAPI
- Uvicorn
- JSON + Base64 image responses

### Frontend

- HTML5
- CSS3
- JavaScript
- Responsive dashboard interface

## API

### Java Backend

Base URL:

`http://localhost:8080`

Health check:

```http
GET /api/health
```

Prediction:

```http
POST /api/predict
Content-Type: multipart/form-data

file=<image>
```

The Java backend forwards the image to the Python AI service and returns the AI response to the browser.

### Python AI Service

Base URL:

`http://127.0.0.1:8000`

Health check:

```http
GET /health
```

Prediction:

```http
POST /predict
Content-Type: multipart/form-data

file=<image>
```

The response contains the prediction status, object-detection decision, class, confidence, gate statistics, and Base64-encoded result images.

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/Dhanush-S-007/Camouflage_Breaker.git
cd Camouflage_Breaker
```

### 2. Create the Python environment

Windows:

```bat
python -m venv venv
venv\Scripts\activate
```

Install Python dependencies:

```bat
pip install -r requirements.txt
```

### 3. Add the dataset

Place COD10K-v3 under:

```text
dataset/
├── Train/
└── Test/
```

The dataset is intentionally excluded from Git.

### 4. Add trained model checkpoints

Large model files are intentionally excluded from Git because of their size.

The inference pipeline expects:

```text
saved_models/
├── sinetv2/
│   └── sinetv2_cod10k_40epoch_best.pth
└── classifier_best.pth
```

The Res2Net checkpoint used by the training/evaluation utilities is expected under:

```text
models/sinetv2/
```

Do not commit large `.pth`, `.pt`, or `.ckpt` files unless the repository strategy is deliberately changed.

### 5. Start the Python AI service

From the project root:

```bat
venv\Scripts\activate
python -m uvicorn ai_service.main:app --host 127.0.0.1 --port 8000
```

Expected service:

```text
http://127.0.0.1:8000
```

### 6. Start the Spring Boot backend

Open a second terminal:

```bat
cd backend
mvnw.cmd spring-boot:run
```

Expected service:

```text
http://localhost:8080
```

### 7. Open the application

Open:

```text
http://localhost:8080/
```

Spring Boot serves the frontend and routes image predictions to the Python AI service.

## Development Notes

The project intentionally uses a two-service architecture:

```text
Browser
   │
   │ HTTP
   ▼
Spring Boot :8080
   │
   │ multipart/form-data
   ▼
FastAPI :8000
   │
   ▼
PyTorch inference
```

This keeps the Java web layer independent from the Python deep-learning stack while allowing the trained models to remain in their native PyTorch environment.

## Version-Control Policy

Model checkpoints, datasets, generated outputs, virtual environments, and build artifacts are excluded from version control.

The repository contains:

- source code
- model architecture code
- training/evaluation scripts
- class mapping
- backend
- AI API
- frontend
- configuration
- documentation

It does not contain:

- COD10K dataset files
- large trained checkpoints
- generated output files
- Python virtual environment
- Maven build output

## Research Foundation

The segmentation component is based on the official SINet-V2 implementation:

**SINet-V2 — Detecting Camouflaged Objects with Pre-trained Models**

Official repository: https://github.com/GewelsJI/SINet-V2

The project uses the official SINet-V2 source architecture while fine-tuning it for the COD10K-based Camouflage Breaker pipeline.

## Public Deployment

For a no-cost public demo, the repository includes a dedicated Streamlit entrypoint:

```text
Internet
   │
   ▼
Streamlit Community Cloud
   │
   ├── Camouflage Breaker UI
   └── SINet-V2 + ResNet50 inference
```

Streamlit Community Cloud supports public GitHub repositories and provides free app hosting. The deployment uses the trained model checkpoints from the GitHub Release instead of storing the large `.pth` files in the source tree.

### Free deployment

1. Open Streamlit Community Cloud.
2. Sign in with GitHub.
3. Choose **Create app**.
4. Select repository: `Dhanush-S-007/Camouflage_Breaker`.
5. Select branch: `main`.
6. Set the entrypoint to `streamlit_app.py`.
7. Use Python 3.11 in **Advanced settings**.
8. Deploy.

The app downloads the two public model checkpoints when the inference pipeline first starts:

```text
SINet-V2 checkpoint: 294 MB
ResNet50 checkpoint: 94.1 MB
```

No local Python, Java, dataset, or model files are required for users of the public app.

### Local vs public deployment

Local development keeps the full production architecture:

```text
Frontend → Spring Boot → FastAPI → SINet-V2 + ResNet50
```

The free public demo uses:

```text
Streamlit Community Cloud → SINet-V2 + ResNet50
```

This deployment variant exists specifically to avoid paid hosting while keeping the same trained inference pipeline and model weights.

### Model files

The trained checkpoints remain outside the normal Git history:

```text
saved_models/
├── sinetv2/
│   └── sinetv2_cod10k_40epoch_best.pth
└── classifier_best.pth
```

The class mapping remains tracked in Git:

```text
saved_models/class_mapping.json
```

The inference pipeline has the public GitHub Release URLs as deployment defaults, while environment variables can still override them for other hosting providers.

## Current Status

- [x] COD10K dataset verification
- [x] SINet-V2 integration
- [x] SINet-V2 fine-tuning
- [x] Full 4,000-image segmentation evaluation
- [x] ResNet50 classifier integration
- [x] 69-class mapping
- [x] Object-presence gate
- [x] End-to-end Python inference pipeline
- [x] FastAPI AI service
- [x] Spring Boot backend
- [x] Professional web frontend
- [x] Java → Python prediction flow
- [x] Result visualization
- [x] Downloadable analysis package
- [x] Git repository cleanup and model-weight exclusion
- [x] Free Streamlit deployment configuration

## Limitations

- Public deployment requires external hosting for the trained model checkpoints.
- Large trained checkpoints are not stored in Git.
- Classification performance should be evaluated separately from the reported segmentation metrics.
- Predictions depend on segmentation quality and the training distribution of the 69-class classifier.
- The COD10K dataset must be obtained and used according to its applicable terms.

## Future Improvements

- Add automated model-download/setup tooling.
- Add dedicated classifier evaluation metrics and confusion matrices.
- Add batch image inference.
- Add GPU deployment configuration.
- Add Docker deployment for the Java and Python services.
- Add automated CI tests for API and frontend integration.
- Add model versioning and experiment tracking.

## Author

**Dhanush S.**

Built as a deep-learning computer vision project focused on practical camouflaged object detection, segmentation, and classification.

## License

The original Camouflage Breaker project code is licensed under the **Apache License 2.0**.

See the root [LICENSE](LICENSE) file for the complete license text.

This repository also contains third-party components, including the SINet-V2 source under `models/sinetv2/source/`. Those components remain subject to their original licenses and attribution requirements. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The project license does not override the license terms of third-party dependencies or datasets.
