# RetinaGuard — Dataset Setup Guide

## Overview

The RetinaGuard application runs in **Demo Mode** when no datasets are connected. This guide explains how to connect the APTOS 2019 and IDRiD datasets to enable real model training and evaluation.

---

## Dataset 1: APTOS 2019 (DR Classification)

### What it contains
- 3,662 retinal images labelled with DR grades 0–4 (ICDR scale)
- Source: Asia Pacific Tele-Ophthalmology Society Blindness Detection competition (Kaggle)

### Download
1. Go to: https://www.kaggle.com/competitions/aptos2019-blindness-detection/data
2. Download:
   - `train.csv`
   - `train_images.zip` (unzip to `train_images/`)
   - `test.csv` (if available)
   - `test_images.zip` (unzip to `test_images/`)

### Placement

```
RetinaGuard/
└── data/
    └── aptos/
        ├── train.csv           ← Required: id_code, diagnosis columns
        ├── train_images/       ← Required: PNG retinal images
        │   ├── 000c1434d8d7.png
        │   ├── 001639a390f0.png
        │   └── ...
        ├── test.csv            ← Optional: for held-out evaluation
        └── test_images/        ← Optional: test set images
```

### CSV Format

`train.csv` must contain:

| id_code | diagnosis |
|---|---|
| 000c1434d8d7 | 2 |
| 001639a390f0 | 4 |
| ... | ... |

- `id_code` — image filename without extension
- `diagnosis` — integer 0–4 (ICDR DR grade)

### Verification

After placement, run in MATLAB:
```matlab
cfg = RGConfig();
T = readtable(fullfile(cfg.paths.aptos, 'train.csv'));
disp(height(T))           % Should show ~3662
tabulate(T.diagnosis)     % Distribution across 5 grades
```

---

## Dataset 2: IDRiD (Lesion Segmentation + Grading)

### What it contains
- 516 retinal images with pixel-level lesion annotations
- Lesion types: microaneurysms, hemorrhages, hard exudates, soft exudates, optic disc
- DR grade labels for each image
- Source: Indian Diabetic Retinopathy Image Dataset

### Download
- Website: https://idrid.grand-challenge.org/
- Or: IEEE DataPort (DOI: 10.21227/H25W98)

### Placement

```
RetinaGuard/
└── data/
    └── idrid/
        ├── B. Disease Grading/
        │   ├── 1. Original Images/
        │   │   ├── a. Training Set/
        │   │   │   ├── IDRiD_001.jpg
        │   │   │   └── ...
        │   │   └── b. Testing Set/
        │   └── 2. Groundtruths/
        │       ├── a. IDRiD_Disease Grading_Training Labels.csv
        │       └── b. IDRiD_Disease Grading_Testing Labels.csv
        └── A. Segmentation/
            ├── 1. Original Images/
            │   ├── a. Training Set/
            │   └── b. Testing Set/
            └── 2. All Segmentation Groundtruths/
                ├── 1. Microaneurysms/
                │   ├── IDRiD_01_MA.tif
                │   └── ...
                ├── 2. Haemorrhages/
                ├── 3. Hard Exudates/
                ├── 4. Soft Exudates/
                └── 5. Optic Disc/
```

### Verification

```matlab
cfg = RGConfig();
imgDir = fullfile(cfg.paths.idrid, 'B. Disease Grading', ...
         '1. Original Images', 'a. Training Set');
d = dir(fullfile(imgDir, '*.jpg'));
fprintf('IDRiD training images found: %d\n', numel(d));
```

---

## Application Detection

When the app starts, it checks dataset paths in `RGConfig.m`:

```matlab
cfg.paths.aptos = fullfile(base, 'data', 'aptos');
cfg.paths.idrid = fullfile(base, 'data', 'idrid');
```

The **Model / AI** panel shows connection status for each dataset.

---

## Train/Validation/Test Split

> **Critical: Never use test labels during training or validation.**

### Recommended split for APTOS 2019

| Split | Fraction | Purpose |
|---|---|---|
| Train | 70% | Model learning |
| Validation | 10% | Hyperparameter tuning, early stopping |
| Test | 20% | Final evaluation (touch only once) |

The training script (`training/train_dr_classifier.m`) uses a fixed random seed (`rng(42)`) for reproducibility.

### IDRiD built-in split
IDRiD provides an official train/test split — **use it as-is**. Do not merge and re-split.

---

## Data Integrity Checks

Before training, run:
```matlab
% Check for corrupted images
cfg = RGConfig();
T   = readtable(fullfile(cfg.paths.aptos, 'train.csv'));
bad = 0;
for k = 1:height(T)
    f = fullfile(cfg.paths.aptos, 'train_images', [T.id_code{k} '.png']);
    try, imfinfo(f); catch, bad = bad+1; end
end
fprintf('Corrupted or missing: %d/%d\n', bad, height(T));
```

---

## Path Configuration

All paths are defined in `config/RGConfig.m`. To change a dataset location:

```matlab
% In RGConfig.m — edit these lines only:
cfg.paths.aptos = fullfile(base, 'data', 'aptos');
cfg.paths.idrid = fullfile(base, 'data', 'idrid');
```

No other files need modification.
