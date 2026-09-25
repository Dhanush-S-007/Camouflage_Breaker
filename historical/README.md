# Historical Development Archive

This directory preserves important development-stage work that led to the current Camouflage Breaker implementation.

These files are **historical**, not part of the production inference path. They are retained so the repository documents how the project evolved rather than hiding earlier approaches.

## Development stages

1. **Initial segmentation — ResUNet**
   - Early ResNet50 encoder + U-Net decoder approach.
   - Preserved under `initial-resunet/`.

2. **Segmentation experiments — SINet-V2 and SAM 2**
   - Earlier SINet-V2 training/evaluation experiments.
   - SAM 2.1 was evaluated as an alternative segmentation route.
   - Preserved under `segmentation-experiments/`.

3. **Classifier development**
   - Earlier classifier training work before the final classifier training pipeline.
   - Preserved under `classifier-development/`.

## Important distinction

The production pipeline does **not** import these historical files.

The current production path is:

```text
COD10K-trained SINet-V2
        ↓
Object Gate
        ↓
Object Crop
        ↓
69-class ResNet50
        ↓
FastAPI
        ↓
Spring Boot
        ↓
Frontend
```

Historical files are retained for reproducibility, project presentation, and development traceability.
