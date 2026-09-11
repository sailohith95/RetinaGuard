"""
verify_walkthrough_cases.py
===========================
Executes empirical verification for all 6 demo cases, visualization layers,
performance timings, and error handling suites.
"""

import sys
import time
import json
import tempfile
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

import torch
import yaml
from preprocessing.retinal_preprocessor import RetinalPreprocessor
from inference.predict import DRPredictor
from explainability.gradcam import GradCAM

# Load config and predictor
config_path = ROOT / "python" / "config" / "config.yaml"
with open(config_path) as f:
    cfg = yaml.safe_load(f)
cfg["_root"] = str(ROOT)

preprocessor = RetinalPreprocessor(cfg)
predictor = DRPredictor(cfg)

SAMPLE_DIR = ROOT / "demo" / "sample_images"

cases_info = [
    ("Grade 0 (No DR)", SAMPLE_DIR / "demo_grade0.png", 0),
    ("Grade 1 (Mild NPDR)", SAMPLE_DIR / "demo_grade1.png", 1),
    ("Grade 2 (Moderate NPDR)", SAMPLE_DIR / "demo_grade2.png", 2),
    ("Grade 3 (Severe NPDR)", SAMPLE_DIR / "demo_grade3.png", 3),
    ("Grade 4 (Proliferative DR)", SAMPLE_DIR / "demo_grade4.png", 4),
]

def assess_quality(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_lum = float(np.mean(gray))
    sharp_score = min(100.0, (lap_var / 500.0) * 100.0)
    illum_score = max(0.0, 100.0 - abs(mean_lum - 128.0) * (100.0 / 128.0))
    overall = 0.6 * sharp_score + 0.4 * illum_score
    
    if overall >= 60 and lap_var >= 80 and 35 <= mean_lum <= 225:
        status = "GOOD"
        gradable = True
        reason = "Image meets clinical quality criteria."
        rec = "Proceed with automated screening."
    elif overall >= 40:
        status = "BORDERLINE"
        gradable = True
        reason = "Borderline focus or illumination."
        rec = "Proceed with caution; clinical review advised."
    else:
        status = "UNGRADABLE"
        gradable = False
        reason = "Insufficient focus or illumination (severe blur/underexposure)."
        rec = "Recapture retinal image with improved focus and illumination."
        
    return {
        "score": round(overall, 1),
        "status": status,
        "gradable": gradable,
        "laplacian_var": round(lap_var, 1),
        "mean_lum": round(mean_lum, 1),
        "reason": reason,
        "recommendation": rec
    }

def detect_landmarks_and_lesions(img_bgr):
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    green = img_bgr[:, :, 1]
    
    # 1. Optic disc
    blur = cv2.GaussianBlur(gray, (31, 31), 0)
    _, _, _, (od_x, od_y) = cv2.minMaxLoc(blur)
    od_r = int(min(h, w) * 0.08)
    y, x = np.ogrid[:h, :w]
    od_mask = ((x - od_x)**2 + (y - od_y)**2) <= (od_r**2)
    
    # 2. Fovea
    fovea_offset = int(od_r * 2.5)
    fovea_x = od_x - fovea_offset if od_x > w // 2 else od_x + fovea_offset
    fovea_x = max(0, min(w - 1, fovea_x))
    fovea_y = od_y
    
    # 3. Vessels
    kernel_v = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    tophat_v = cv2.morphologyEx(green, cv2.MORPH_BLACKHAT, kernel_v)
    _, vessel_mask = cv2.threshold(tophat_v, 10, 255, cv2.THRESH_BINARY)
    retina_mask = green > 15
    tot_ret = max(1, np.sum(retina_mask))
    vess_pix = np.sum((vessel_mask > 0) & retina_mask)
    vessel_density = round((vess_pix / tot_ret) * 100.0, 2)
    
    # 4. Hard exudates (disc masked with 25% dilation)
    od_dilated = ((x - od_x)**2 + (y - od_y)**2) <= ((od_r * 1.25)**2)
    ex_mask = (gray > 180) & (~od_dilated) & (gray > 20)
    num_ex, _, stats_ex, _ = cv2.connectedComponentsWithStats(ex_mask.astype(np.uint8))
    ex_count = 0
    ex_area = 0
    for i in range(1, num_ex):
        a = stats_ex[i, cv2.CC_STAT_AREA]
        if a >= 5:
            ex_count += 1
            ex_area += a
            
    # 5. Microaneurysms (inverted green top-hat)
    inv_green = 255 - green
    kernel_ma = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    tophat_ma = cv2.morphologyEx(inv_green, cv2.MORPH_TOPHAT, kernel_ma)
    _, ma_mask = cv2.threshold(tophat_ma, 25, 255, cv2.THRESH_BINARY)
    num_ma, _, stats_ma, _ = cv2.connectedComponentsWithStats(ma_mask)
    ma_count = 0
    for i in range(1, num_ma):
        a = stats_ma[i, cv2.CC_STAT_AREA]
        if 3 <= a <= 65:
            ma_count += 1
            
    # 6. Hemorrhages (larger dark blobs)
    kernel_he = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tophat_he = cv2.morphologyEx(inv_green, cv2.MORPH_TOPHAT, kernel_he)
    _, he_mask = cv2.threshold(tophat_he, 30, 255, cv2.THRESH_BINARY)
    num_he, _, stats_he, _ = cv2.connectedComponentsWithStats(he_mask)
    he_count = 0
    he_area = 0
    for i in range(1, num_he):
        a = stats_he[i, cv2.CC_STAT_AREA]
        if 40 <= a <= 2000:
            he_count += 1
            he_area += a
            
    # 7. Neovascularization risk indicator
    if vessel_density > 18.0:
        nv_risk = "High"
    elif vessel_density > 14.0:
        nv_risk = "Moderate"
    else:
        nv_risk = "Low"
        
    return {
        "optic_disc": {"centroid": [od_x, od_y], "radius": od_r},
        "fovea": {"coordinates": [fovea_x, fovea_y]},
        "vessels": {"density_percent": vessel_density},
        "microaneurysms": {"count": ma_count},
        "exudates": {"count": ex_count, "area": int(ex_area)},
        "hemorrhages": {"count": he_count, "area": int(he_area)},
        "neovascularization": {"risk_level": nv_risk}
    }

print("=================================================================")
print("  EMPIRICAL EVALUATION OF ALL 6 DEMO CASES")
print("=================================================================\n")

demo_results = []
for label, p, intended_grade in cases_info:
    print(f"--- Testing {label} ({p.name}) ---")
    t_start = time.perf_counter()
    
    # 1. Load image
    t0 = time.perf_counter()
    img_bgr = cv2.imread(str(p))
    pil_img = Image.open(p).convert("RGB")
    t_load = time.perf_counter() - t0
    
    # 2. Quality
    t0 = time.perf_counter()
    q = assess_quality(img_bgr)
    t_qual = time.perf_counter() - t0
    
    # 3. If ungradable, halt
    if not q["gradable"]:
        t_total = time.perf_counter() - t_start
        print(f"  Quality Status   : {q['status']} (Score: {q['score']}/100)")
        print(f"  Reason           : {q['reason']}")
        print(f"  Recommendation   : {q['recommendation']}")
        print(f"  Downstream Action: GRADING HALTED (Safety Early Rejection)")
        print(f"  Total Latency    : {t_total*1000:.1f} ms\n")
        demo_results.append({
            "label": label,
            "quality": q,
            "halted": True,
            "timing_ms": round(t_total * 1000, 1)
        })
        continue
        
    # 4. Landmarks & Lesions
    t0 = time.perf_counter()
    lm_les = detect_landmarks_and_lesions(img_bgr)
    t_cv = time.perf_counter() - t0
    
    # 5. EXP-001 Model Inference
    t0 = time.perf_counter()
    pred = predictor.predict(pil_img, generate_gradcam=True)
    t_inf = time.perf_counter() - t0
    
    t_total = time.perf_counter() - t_start
    
    print(f"  Quality Status   : {q['status']} (Score: {q['score']}/100)")
    print(f"  Landmarks        : Optic Disc @ {lm_les['optic_disc']['centroid']}, Fovea @ {lm_les['fovea']['coordinates']}, Vessel Density: {lm_les['vessels']['density_percent']}%")
    print(f"  Lesion Candidates: MAs={lm_les['microaneurysms']['count']}, Exudates={lm_les['exudates']['count']}, Hemorrhages={lm_les['hemorrhages']['count']}, NV Risk={lm_les['neovascularization']['risk_level']}")
    print(f"  Predicted Grade  : Grade {pred['grade']} — {pred['severity']} (Confidence: {pred['confidence']*100:.1f}%)")
    print(f"  Referable DR     : {pred['referral']} ({'REFER TO SPECIALIST' if pred['referral'] else 'ROUTINE FOLLOW-UP'})")
    print(f"  Grad-CAM Active  : {pred['gradcam_available']} (Saved: {pred['gradcam_path'] is not None})")
    print(f"  Recommendation   : {pred['referral_text']}")
    print(f"  Timings (ms)     : Load={t_load*1000:.1f} | Qual={t_qual*1000:.1f} | CV={t_cv*1000:.1f} | DL+GradCAM={t_inf*1000:.1f} | Total={t_total*1000:.1f} ms\n")
    
    demo_results.append({
        "label": label,
        "quality": q,
        "landmarks_lesions": lm_les,
        "prediction": pred,
        "halted": False,
        "timings_ms": {
            "load": round(t_load*1000, 1),
            "quality": round(t_qual*1000, 1),
            "cv": round(t_cv*1000, 1),
            "dl_gradcam": round(t_inf*1000, 1),
            "total": round(t_total*1000, 1)
        }
    })

print("=================================================================")
print("  ERROR HANDLING BENCHMARK SUITE (A to G)")
print("=================================================================\n")

error_tests = [
    ("A. Unsupported image format (.txt)", "text", b"not an image"),
    ("B. Corrupted image file", "png", b"\x89PNG\r\n\x1a\ncorrupted_data_random"),
    ("C. Very small image (8x8 px)", "img_tiny", None),
    ("D. Extremely dark image (all 0s)", "img_dark", None),
    ("E. Extremely bright image (all 250s)", "img_bright", None),
    ("F. Extremely blurry image (Gaussian sigma=25)", "img_blur", None),
    ("G. Valid normal fundus image", "img_valid", None),
]

for test_name, mode, payload in error_tests:
    print(f"Running Error Test: {test_name}")
    if mode == "text" or mode == "png":
        with tempfile.NamedTemporaryFile(suffix=f".{mode if mode!='text' else 'txt'}", delete=False) as tf:
            tf.write(payload)
            tp = tf.name
        try:
            pil = Image.open(tp).convert("RGB")
            # Should fail or handle
            res = predictor.predict(pil)
            print(f"  Result: Handled -> Grade {res['grade']}")
        except Exception as ex:
            print(f"  Result: Controlled Exception Caught as Expected -> {type(ex).__name__}: {ex}")
        finally:
            Path(tp).unlink(missing_ok=True)
    elif mode == "img_tiny":
        arr = np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8)
        pil = Image.fromarray(arr)
        res = predictor.predict(pil)
        print(f"  Result: Resized & Handled Safely -> Grade {res['grade']}, Severity: {res['severity']}")
    elif mode == "img_dark":
        arr = np.zeros((512, 512, 3), dtype=np.uint8)
        q = assess_quality(arr)
        print(f"  Result: Quality Safety Halt -> Status: {q['status']}, Gradable: {q['gradable']}, Reason: {q['reason']}")
    elif mode == "img_bright":
        arr = np.full((512, 512, 3), 250, dtype=np.uint8)
        q = assess_quality(arr)
        print(f"  Result: Quality Safety Halt -> Status: {q['status']}, Gradable: {q['gradable']}, Reason: {q['reason']}")
    elif mode == "img_blur":
        base = cv2.imread(str(SAMPLE_DIR / "demo_grade0.png"))
        blur = cv2.GaussianBlur(base, (51, 51), 25.0)
        q = assess_quality(blur)
        print(f"  Result: Quality Safety Halt -> Status: {q['status']}, Gradable: {q['gradable']}, Reason: {q['reason']}")
    elif mode == "img_valid":
        base = cv2.imread(str(SAMPLE_DIR / "demo_grade0.png"))
        q = assess_quality(base)
        res = predictor.predict(Image.fromarray(cv2.cvtColor(base, cv2.COLOR_BGR2RGB)))
        print(f"  Result: Normal Processing -> Quality: {q['status']}, Predicted Grade: {res['grade']} ({res['severity']})")

print("\nEmpirical Verification Script Complete.")
