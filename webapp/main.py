"""
main.py
=======
FastAPI Application for RetinaGuard AI-Assisted Screening Workstation.
"""

import webapp  # Applies Starlette/FastAPI compatibility shim
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from webapp.services.screening_service import ScreeningService

ROOT = Path(__file__).resolve().parent.parent
WEBAPP_DIR = ROOT / "webapp"
UPLOADS_DIR = WEBAPP_DIR / "uploads"
OUTPUTS_DIR = WEBAPP_DIR / "outputs"
DEMO_DIR = ROOT / "demo" / "sample_images"

# Ensure directories exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="RetinaGuard — AI-Assisted Retinal Screening Workstation",
    description="Standalone Localhost Web Workstation for SIH 2026 Diabetic Retinopathy Screening",
    version="1.0.0"
)

# Mount static file directories
app.mount("/static", StaticFiles(directory=str(WEBAPP_DIR / "static")), name="static")
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")
if DEMO_DIR.exists():
    app.mount("/demo/sample_images", StaticFiles(directory=str(DEMO_DIR)), name="demo_images")

templates = Jinja2Templates(directory=str(WEBAPP_DIR / "templates"))

# Initialize screening orchestrator
screening_service = ScreeningService(uploads_dir=UPLOADS_DIR, outputs_dir=OUTPUTS_DIR)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serves the main clinical workstation single-page interface."""
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/health")
async def health():
    """Health check endpoint confirming production model and offline status."""
    return {
        "status": "ok",
        "model": "EXP-001",
        "architecture": "EfficientNet-B0",
        "offline": True,
        "qwk": 0.7342,
        "accuracy": 0.6985,
        "referable_sensitivity": 0.7886,
        "referable_specificity": 0.9264
    }


@app.get("/api/demo-cases")
async def get_demo_cases():
    """Returns the list of 6 curated authentic demo screening cases."""
    return screening_service.get_demo_cases()


@app.post("/api/analyze")
async def analyze_image(
    file: Optional[UploadFile] = File(None),
    demo_id: Optional[str] = Form(None),
    patient_id: Optional[str] = Form("PT-82910"),
    exam_id: Optional[str] = Form("EX-00412"),
    eye: Optional[str] = Form("Right (OD)")
):
    """
    Core screening endpoint: accepts an uploaded fundus scan or demo case ID,
    executes quality assessment, enhancement, structures, lesions, EXP-001 grading,
    Grad-CAM, recommendations, and report generation.
    """
    target_path = None
    original_url = None
    demo_ref = None

    if demo_id:
        cases = {c["id"]: c for c in screening_service.get_demo_cases()}
        # Resolve ID (supporting grade_0..grade_4 as primary, and legacy case_1..case_5 aliases)
        alias_map = {
            "case_1": "grade_0",
            "case_2": "grade_1",
            "case_3": "grade_2",
            "case_4": "grade_3",
            "case_5": "grade_4"
        }
        resolved_id = alias_map.get(demo_id, demo_id)
        if resolved_id not in cases:
            raise HTTPException(status_code=400, detail=f"Unknown demo grade ID: {demo_id}")
        case_info = cases[resolved_id]
        target_path = DEMO_DIR / case_info["filename"]
        original_url = case_info["image_url"]
        demo_ref = {
            "id": case_info["id"],
            "label": case_info["label"],
            "reference_grade": case_info["reference_grade"],
            "reference_name": case_info["reference_name"]
        }
    elif file and file.filename:
        # Validate extension
        ext = Path(file.filename).suffix.lower()
        if ext not in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid file format. Supported retinal image formats are JPG, JPEG, PNG, and TIFF."
            )

        # Save uploaded file
        clean_filename = f"upload_{exam_id}_{Path(file.filename).name}"
        save_path = UPLOADS_DIR / clean_filename
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Verify file size > 0
        if save_path.stat().st_size == 0:
            save_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=400,
                detail="Empty file uploaded. Please provide a valid fundus image."
            )

        target_path = save_path
        original_url = f"/uploads/{clean_filename}"
    else:
        raise HTTPException(
            status_code=400,
            detail="No fundus image provided. Upload an image file or select a built-in demo case."
        )

    try:
        result = screening_service.run_screening(
            image_path=target_path,
            original_url=original_url,
            patient_id=patient_id,
            exam_id=exam_id,
            eye=eye,
            demo_reference=demo_ref
        )
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Screening analysis failed: {str(e)}"
        )


@app.get("/api/download-report/{filename}")
async def download_report(filename: str):
    """Direct download endpoint for generated clinical reports."""
    report_file = OUTPUTS_DIR / filename
    if not report_file.exists():
        raise HTTPException(status_code=404, detail="Report file not found.")
    return FileResponse(
        path=str(report_file),
        filename=filename,
        media_type="text/html"
    )
