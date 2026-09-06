# RetinaGuard — Models Directory

This directory contains trained MATLAB model files.

## Structure

```
models/
├── dr/
│   └── dr_classifier.mat      ← APTOS-trained DR severity classifier
│                                  Variables: net (dlnetwork), classes (categorical)
├── quality/
│   └── quality_model.mat      ← Image quality assessment model (optional)
├── lesions/
│   └── lesion_model.mat       ← IDRiD-trained lesion segmentation model
└── structures/
    └── structure_model.mat    ← Optic disc/fovea/vessel detection model
```

## Status

**No trained models are currently installed.**

The application uses image-processing heuristics (demo mode) when models are absent.

See [MODEL_TRAINING.md](../MODEL_TRAINING.md) for training instructions.

## Model Loading

The app checks for each `.mat` file at startup.
- If found → **Trained Model** mode
- If missing → **Demo Mode** (clearly labelled in UI)

No code changes are required when models are added.
