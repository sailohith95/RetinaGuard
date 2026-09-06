# RetinaGuard — Model Training Guide

## Overview

This guide explains how to train the DR severity classifier and connect it to the RetinaGuard application. Once the model file is saved, the app **automatically switches from Demo Mode to Trained Model mode** — no code changes required.

---

## DR Severity Classifier

### Approach: Transfer Learning

Training a deep CNN from scratch on 3,662 images is inadvisable due to overfitting risk. Instead, RetinaGuard uses **fine-tuning** of a pretrained ImageNet network:

```
ImageNet Pretrained CNN (GoogLeNet / ResNet-50 / EfficientNet-B0)
    ↓
Replace final fully-connected layer → 5-class output (Grades 0–4)
    ↓
Freeze early layers, train classification head first
    ↓
Unfreeze all layers, fine-tune with low learning rate
    ↓
Save best checkpoint (minimum validation loss)
    ↓
models/dr/dr_classifier.mat
```

### Prerequisites

1. MATLAB Deep Learning Toolbox
2. One of:
   - **Deep Learning Toolbox Model for GoogLeNet Network** (recommended for speed)
   - **Deep Learning Toolbox Model for ResNet-50 Network**
3. APTOS 2019 dataset in `data/aptos/` (see `DATASET_SETUP.md`)
4. GPU strongly recommended (NVIDIA with CUDA); CPU training will take many hours

### Run Training

```matlab
cd('C:\Users\T SAILOHITH\Desktop\SIH\RetinaGuard')
addpath(genpath('.'))
run('training/train_dr_classifier.m')
```

The script automatically:
- Reads `data/aptos/train.csv`
- Loads images from `data/aptos/train_images/`
- Splits 80% train / 20% validation
- Applies data augmentation (flips, rotation ±30°, translation)
- Fine-tunes GoogLeNet with Adam optimizer, lr=1e-4
- Saves best model to `models/dr/dr_classifier.mat`

### Estimated Training Time

| Hardware | Approximate Time |
|---|---|
| NVIDIA GPU (RTX 3060+) | 1–3 hours |
| NVIDIA GPU (GTX 1060) | 3–8 hours |
| CPU only | 12–48 hours |

### Connecting the Model to the App

After training completes:

```
models/dr/dr_classifier.mat   ← saved automatically by training script
```

The file contains:
- `net` — trained `dlnetwork` or `SeriesNetwork`
- `classes` — `categorical([0 1 2 3 4])`

**Restart** `RetinaGuardApp` — it detects and loads the model automatically. The Demo Mode banner disappears and the Model / AI panel shows "Trained Model Loaded".

---

## Choosing a Base Network

| Network | MATLAB function | Size | Speed | Accuracy (typical) |
|---|---|---|---|---|
| GoogLeNet | `googlenet` | 27 MB | Fast | Good baseline |
| ResNet-50 | `resnet50` | 98 MB | Medium | Better |
| EfficientNet-B0 | `efficientnetb0` | 20 MB | Fast | Competitive |

To switch networks, edit `training/train_dr_classifier.m` — change `googlenet` to `resnet50` or `efficientnetb0` and update the layer names for the replacement block accordingly.

---

## Class Imbalance

APTOS 2019 has significant class imbalance:

| Grade | Approximate count |
|---|---|
| 0 (No DR) | ~1805 |
| 1 (Mild) | ~370 |
| 2 (Moderate) | ~999 |
| 3 (Severe) | ~193 |
| 4 (PDR) | ~295 |

Recommended strategies:
- **Weighted loss** — assign higher loss weight to underrepresented classes
- **Oversampling** — duplicate rare class examples during augmentation
- **Class-balanced sampling** — use `BalancedDiscreteSampler` if available

Add class weights to `classificationLayer`:
```matlab
classWeights = 1 ./ classCount;
classWeights = classWeights / sum(classWeights) * numClasses;
newOutput = classificationLayer('Name','output_dr','Classes',classes,'ClassWeights',classWeights);
```

---

## Lesion Segmentation Model (IDRiD)

The lesion detection module (`core/lesions/detectLesions.m`) currently uses colour thresholding heuristics. To replace with a trained segmentation model:

### Recommended Architecture
- **U-Net** (MATLAB Deep Learning Toolbox supports `unetLayers`)
- One model per lesion type OR a multi-class segmentation model

### Training Outline
```matlab
% Per lesion type (e.g., microaneurysms):
lgraph = unetLayers([512 512 3], 2, 'EncoderDepth', 4);
% Load IDRiD images + binary masks
% Train with weighted cross-entropy (foreground rare)
% Save: save(cfg.paths.lesionModel, 'netMA', 'netExudate', ...)
```

### Connecting
In `detectLesions.m`, add at the top:
```matlab
if isfile(cfg.paths.lesionModel)
    data = load(cfg.paths.lesionModel);
    % Run semantic segmentation with data.netMA etc.
    result.demoMode = false;
    return;
end
% ... existing heuristic code continues as fallback
```

---

## Structure Detection Model

The structure detection module (`core/structures/detectStructures.m`) uses geometric heuristics for the optic disc and fovea. For improved accuracy:

- **Optic Disc**: train a U-Net segmentation on IDRiD optic disc masks
- **Fovea**: train a regression CNN predicting fovea (x,y) coordinates
- **Vessels**: MATLAB's `vesselSeg` (if available) or a trained U-Net on DRIVE/CHASEDB dataset

---

## Model Versioning

When retraining, use dated model filenames and copy the best to the standard path:

```
models/dr/
├── dr_classifier.mat           ← active model (loaded by app)
├── dr_classifier_v1_20260905.mat
└── dr_classifier_v2_20261201.mat
```

Track model versions alongside dataset version and training config used.

---

## Important Notes

> **Do not report training-set or validation-set metrics as final performance.**
> Only report metrics computed on the held-out test set.
> See `EVALUATION.md` for the correct evaluation procedure.

> **Demo Mode remains available regardless of model state.**
> If `dr_classifier.mat` is deleted or corrupted, the app gracefully falls back to Demo Mode.
