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
def analyze_image(
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
    Executed as synchronous 'def' to offload CPU-bound inference to Starlette threadpool,
    preventing event loop starvation.
    """
    import time
    print("[SCREENING] Image Upload/Read START")
    t_upload_0 = time.time()
    target_path = None
    original_url = None
    demo_ref = None

    if demo_id:
        cases = {c["id"]: c for c in screening_service.get_demo_cases()}
        if demo_id not in cases:
            raise HTTPException(status_code=400, detail=f"Unknown demo case ID: {demo_id}")
        case_info = cases[demo_id]
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

    t_upload_1 = time.time()
    print(f"[SCREENING] Image Upload/Read COMPLETE {t_upload_1 - t_upload_0:.2f}s")

    try:
        result = screening_service.run_screening(
            image_path=target_path,
            original_url=original_url,
            patient_id=patient_id,
            exam_id=exam_id,
            eye=eye,
            demo_reference=demo_ref
        )
        print("[SCREENING] Response Serialization START")
        t_ser_0 = time.time()
        resp = JSONResponse(content=result)
        print(f"[SCREENING] Response Serialization COMPLETE {time.time() - t_ser_0:.2f}s")
        return resp
    except HTTPException:
        raise
    except Exception as e:
        print(f"[SCREENING] ERROR in analyze_image: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Screening analysis failed: {str(e)}"
        )


@app.post("/api/gradcam")
async def compute_gradcam(
    request: Request,
    image_url: Optional[str] = Form(None),
    demo_id: Optional[str] = Form(None),
    target_grade: Optional[int] = Form(None)
):
    """
    Dedicated on-demand explainability endpoint.
    Accepts form-data, JSON, or query parameters.
    Executes REAL PyTorch Grad-CAM with server-side timeout guard.
    Does not block or affect the primary screening diagnosis.
    """
    # Check if parameters provided in JSON body or Query
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            body = await request.json()
            if isinstance(body, dict):
                demo_id = demo_id or body.get("demo_id")
                image_url = image_url or body.get("image_url")
                if target_grade is None and "target_grade" in body:
                    try:
                        target_grade = int(body["target_grade"])
                    except (ValueError, TypeError):
                        pass
        except Exception:
            pass

    if not demo_id and "demo_id" in request.query_params:
        demo_id = request.query_params["demo_id"]
    if not image_url and "image_url" in request.query_params:
        image_url = request.query_params["image_url"]
    if target_grade is None and "target_grade" in request.query_params:
        try:
            target_grade = int(request.query_params["target_grade"])
        except (ValueError, TypeError):
            pass
    target_path = None
    if demo_id:
        cases = {c["id"]: c for c in screening_service.get_demo_cases()}
        if demo_id not in cases:
            raise HTTPException(status_code=400, detail=f"Unknown demo case ID: {demo_id}")
        target_path = DEMO_DIR / cases[demo_id]["filename"]
        if target_grade is None:
            ref_g = cases[demo_id].get("reference_grade", 0)
            target_grade = max(0, int(ref_g)) if ref_g is not None else 0
    elif image_url:
        clean_rel = image_url.lstrip("/")
        if clean_rel.startswith("uploads/"):
            filename = clean_rel.replace("uploads/", "", 1)
            target_path = UPLOADS_DIR / filename
        elif clean_rel.startswith("demo/sample_images/"):
            filename = clean_rel.replace("demo/sample_images/", "", 1)
            target_path = DEMO_DIR / filename
        else:
            p_up = UPLOADS_DIR / Path(clean_rel).name
            p_dm = DEMO_DIR / Path(clean_rel).name
            target_path = p_up if p_up.exists() else p_dm
    else:
        raise HTTPException(status_code=400, detail="Missing image_url or demo_id parameter.")

    if not target_path or not target_path.exists():
        raise HTTPException(status_code=404, detail="Referenced retinal image file not found on server.")

    if target_grade is None:
        target_grade = 0

    try:
        gradcam_result = screening_service.run_gradcam(
            image_path=target_path,
            target_grade=int(target_grade),
            timeout_sec=20.0
        )
        return JSONResponse(content=gradcam_result)
    except Exception as e:
        return JSONResponse(
            status_code=200,
            content={
                "gradcam_available": False,
                "error": f"Grad-CAM explainability failed: {str(e)}",
                "gradcam_url": None,
                "disclaimer": "Explainability visualization temporarily unavailable on this server."
            }
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
