# RetinaGuard — Web Deployment & Operations Guide

> **Deploying the RetinaGuard AI-Assisted Diabetic Retinopathy Screening Web Application**  
> *Instructions for local execution, Docker/Linux environments, and public Render cloud deployment.*

---

## 1. System Architecture Overview

RetinaGuard provides an edge-deployable, browser-based clinical screening workstation powered by **FastAPI** and the locked production **EXP-001** model (`EfficientNet-B0` exported to `models/aptos_efficientnet/best_model.onnx`).

- **Application Module**: `webapp.main:app`
- **Launcher**: `python webapp/run.py`
- **Production Model**: `models/aptos_efficientnet/best_model.onnx` (Config-relative path resolution via `pathlib`)
- **Execution Engine**: ONNX Runtime (CPU Execution Provider) with PyTorch Grad-CAM explainability
- **Health Check**: `GET /health` (Returns `{"status": "ok", "model": "EXP-001", ...}`)
- **Security & Privacy**: Zero cloud telemetry, stateless image handling, full air-gap compliance

---

## 2. Localhost Execution (Single Command)

To run RetinaGuard locally on your workstation:

```bash
# 1. Install production dependencies
pip install -r requirements.txt

# 2. Launch the workstation
python webapp/run.py
```

The launcher will:
- Bind to `0.0.0.0` (accessible via `http://localhost:8000`).
- Check port availability with dynamic fallback if `8000` is busy.
- Automatically launch your default browser to `http://localhost:8000`.

---

## 3. Public Web Deployment on Render (Render.com)

RetinaGuard includes an automated infrastructure-as-code specification (`render.yaml`) for one-click deployment on Render.

### Option A: Automated Blueprint Deployment
1. Push this repository to GitHub (see Section 4).
2. Log in to [Render Dashboard](https://dashboard.render.com/).
3. Click **New +** → **Blueprint**.
4. Connect your GitHub repository.
5. Render reads `render.yaml` and deploys automatically.

### Option B: Manual Web Service Setup
If creating a **New Web Service** manually on Render:

| Configuration Field | Value |
| :--- | :--- |
| **Name** | `retinaguard` |
| **Language / Runtime** | `Python` |
| **Branch** | `main` (or `master`) |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `uvicorn webapp.main:app --host 0.0.0.0 --port $PORT` |
| **Health Check Path** | `/health` |

#### Required Environment Variables
Add these in the **Environment** settings tab:
- `PYTHON_VERSION`: `3.11.9` (matches `.python-version`)
- `PORT`: `8000` (Render overrides this dynamically with its assigned port)
- `HOST`: `0.0.0.0`

---

## 4. GitHub Publishing Steps

Before pushing to a public repository, verify that dataset caches and local checkpoints are excluded by `.gitignore`:

```bash
# 1. Stage all deployment-ready files
git add .

# 2. Verify staged files (ensure data/ and private checkpoints are NOT staged)
git status

# 3. Create initial commit
git commit -m "feat: RetinaGuard production web workstation and ONNX inference pipeline"

# 4. Connect to your GitHub repository
git remote add origin https://github.com/<YOUR_USERNAME>/RetinaGuard.git
git branch -M main
git push -u origin main
```

---

## 5. Pre-Deployment Verification Checklist

Before publishing your deployment URL, confirm these checks pass:

- [x] **Production Model**: `models/aptos_efficientnet/best_model.onnx` and `best_model.onnx.data` are tracked.
- [x] **No Hardcoded Absolute Paths**: All paths in `webapp/` and `python/` use `pathlib.Path` relative to project root.
- [x] **Headless Compatibility**: Using `opencv-python-headless` to eliminate missing X11/libGL shared library errors on Linux.
- [x] **Ungradable Safety Gate**: Rejects ungradable images (Score < 40/100) and halts grading.
- [x] **Clinical Disclaimer**: Stated prominently on the UI and in clinical reports.
- [x] **Zero Dataset Leakage**: `data/aptos_cache/` and competition images are strictly excluded via `.gitignore`.
