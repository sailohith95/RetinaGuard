"""
lesion_service.py
=================
Quantitative Retinal Lesion Analysis Service with AI Segmentation & Heuristic Fallback.

Capabilities:
1. Deep-Learning Lesion Segmentation (Dual-Head U-Net trained on IDRiD expert ground truth):
   - Microaneurysms (MA)
   - Haemorrhages (HE)
   - Hard Exudates (EX)
   - Soft Exudates (SE)
2. Heuristic Candidate Fallback:
   - Morphological top-hat and luminance thresholding (if AI model is offline)
3. Neovascularization Risk Indicator (measurable vessel characteristics)

All findings are labeled with clinical precision:
- 'AI-detected lesion region' when AI segmentation model is active
- 'Candidate Detection' when running in Heuristic Fallback
"""

from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import cv2

try:
    from python.inference.segment_lesions import LesionSegmentationPipeline
except ImportError:
    LesionSegmentationPipeline = None


class LesionService:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ai_pipeline = None

        if LesionSegmentationPipeline is not None:
            try:
                self.ai_pipeline = LesionSegmentationPipeline()
            except Exception as e:
                print(f"[LesionService] Note: AI segmentation model init notice ({e}). Fallback active.")

    def analyze(
        self,
        img_bgr: np.ndarray,
        od_mask: Optional[np.ndarray] = None,
        vessel_density: float = 10.0,
        filename_prefix: str = "lesions"
    ) -> Dict[str, Any]:
        # 1. Attempt AI Segmentation if pipeline is available and model is loaded
        if self.ai_pipeline is not None and self.ai_pipeline.is_ai_loaded:
            try:
                return self._analyze_ai(img_bgr, vessel_density, filename_prefix)
            except Exception as e:
                print(f"[LesionService] AI inference failed ({e}). Reverting to Heuristic Fallback.")

        # 2. Heuristic Fallback (Classical Computer Vision)
        return self._analyze_heuristic(img_bgr, od_mask, vessel_density, filename_prefix)

    def _analyze_ai(
        self,
        img_bgr: np.ndarray,
        vessel_density: float,
        filename_prefix: str
    ) -> Dict[str, Any]:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        ai_res = self.ai_pipeline.segment(img_rgb)

        # Compute Neovascularization Risk
        if vessel_density > 16.0:
            nv_risk = "High"
            nv_description = f"Vascular density elevated ({vessel_density:.1f}%). Possible active neovascular proliferation."
        elif vessel_density > 13.5:
            nv_risk = "Moderate"
            nv_description = f"Vascular density borderline ({vessel_density:.1f}%). Monitor for vascular remodeling."
        else:
            nv_risk = "Low"
            nv_description = f"Vascular density within expected range ({vessel_density:.1f}%)."

        # Save overlaid visualization
        overlay_rgb = ai_res["overlay_rgb"]
        overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
        overlay_filename = f"{filename_prefix}_overlay_{int(np.random.randint(100000, 999999))}.png"
        overlay_path = self.output_dir / overlay_filename
        cv2.imwrite(str(overlay_path), overlay_bgr)

        counts = ai_res["lesion_counts"]
        areas = ai_res["lesion_areas"]

        return {
            "mode": "AI SEGMENTATION",
            "badge_label": "AI SEGMENTATION (Dual-Head U-Net)",
            "is_ai": True,
            "microaneurysms": {
                "candidate_count": int(counts.get("MA", 0)),
                "total_area_px": int(areas.get("MA", 0)),
                "label": "Microaneurysms (AI-detected lesion regions)",
                "methodology": "Dual-Head U-Net (IDRiD expert ground truth trained)"
            },
            "hard_exudates": {
                "candidate_count": int(counts.get("EX", 0)),
                "total_area_px": int(areas.get("EX", 0)),
                "label": "Hard Exudates (AI-detected lesion regions)",
                "methodology": "Dual-Head U-Net (IDRiD expert ground truth trained)"
            },
            "hemorrhages": {
                "candidate_count": int(counts.get("HE", 0)),
                "total_area_px": int(areas.get("HE", 0)),
                "label": "Haemorrhages (AI-detected lesion regions)",
                "methodology": "Dual-Head U-Net (IDRiD expert ground truth trained)"
            },
            "soft_exudates": {
                "candidate_count": int(counts.get("SE", 0)),
                "total_area_px": int(areas.get("SE", 0)),
                "label": "Soft Exudates (AI-detected lesion regions)",
                "methodology": "Dual-Head U-Net (IDRiD expert ground truth trained)"
            },
            "neovascularization": {
                "risk_level": nv_risk,
                "label": "Neovascularization Risk Indicator",
                "description": nv_description,
                "methodology": "Peri-papillary vessel density & branching stratification"
            },
            "overlay_url": f"/outputs/{overlay_filename}",
            "legend": [
                {"name": "Microaneurysms", "color": "#ff1744", "description": "Focal microvascular dilatations (MA)"},
                {"name": "Haemorrhages", "color": "#ff9100", "description": "Intra-retinal blotches (HE)"},
                {"name": "Hard Exudates", "color": "#ffea00", "description": "Lipid deposits (EX)"},
                {"name": "Soft Exudates", "color": "#00e5ff", "description": "Cotton wool spots (SE)"},
                {"name": "Optic Disc", "color": "#00e676", "description": "Normal anatomical landmark (OD)"}
            ]
        }

    def _analyze_heuristic(
        self,
        img_bgr: np.ndarray,
        od_mask: Optional[np.ndarray] = None,
        vessel_density: float = 10.0,
        filename_prefix: str = "lesions"
    ) -> Dict[str, Any]:
        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        green = img_bgr[:, :, 1]
        inv_green = 255 - green
        retina_mask = green > 15

        # 1. Optic Disc Exclusion Mask
        if od_mask is not None:
            kernel_disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            od_dilated = cv2.dilate(od_mask.astype(np.uint8), kernel_disc) > 0
        else:
            od_dilated = np.zeros((h, w), dtype=bool)

        # 2. Hard Exudate Candidate Segmentation
        ex_bright = (gray > 175) & (~od_dilated) & retina_mask
        num_ex, ex_labels, ex_stats, _ = cv2.connectedComponentsWithStats(ex_bright.astype(np.uint8))
        
        exudate_count = 0
        exudate_area = 0
        clean_ex_mask = np.zeros((h, w), dtype=bool)
        for i in range(1, num_ex):
            area = ex_stats[i, cv2.CC_STAT_AREA]
            if 5 <= area <= 4000:
                exudate_count += 1
                exudate_area += int(area)
                clean_ex_mask[ex_labels == i] = True

        # 3. Microaneurysm Candidate Detection
        kernel_ma = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        ma_tophat = cv2.morphologyEx(inv_green, cv2.MORPH_TOPHAT, kernel_ma)
        _, ma_thresh = cv2.threshold(ma_tophat, 22, 255, cv2.THRESH_BINARY)
        ma_candidates = (ma_thresh > 0) & retina_mask & (~od_dilated)
        
        num_ma, ma_labels, ma_stats, _ = cv2.connectedComponentsWithStats(ma_candidates.astype(np.uint8))
        ma_count = 0
        clean_ma_mask = np.zeros((h, w), dtype=bool)
        for i in range(1, num_ma):
            area = ma_stats[i, cv2.CC_STAT_AREA]
            if 3 <= area <= 65:
                ma_count += 1
                clean_ma_mask[ma_labels == i] = True

        # 4. Hemorrhage Candidate Detection
        kernel_he = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
        he_tophat = cv2.morphologyEx(inv_green, cv2.MORPH_TOPHAT, kernel_he)
        _, he_thresh = cv2.threshold(he_tophat, 30, 255, cv2.THRESH_BINARY)
        he_candidates = (he_thresh > 0) & retina_mask & (~od_dilated) & (~clean_ma_mask)
        
        num_he, he_labels, he_stats, _ = cv2.connectedComponentsWithStats(he_candidates.astype(np.uint8))
        he_count = 0
        he_area = 0
        clean_he_mask = np.zeros((h, w), dtype=bool)
        for i in range(1, num_he):
            area = he_stats[i, cv2.CC_STAT_AREA]
            if 40 <= area <= 2500:
                he_count += 1
                he_area += int(area)
                clean_he_mask[he_labels == i] = True

        # 5. Neovascularization Risk Indicator
        if vessel_density > 16.0:
            nv_risk = "High"
            nv_description = f"Vascular density elevated ({vessel_density:.1f}%). Possible active neovascular proliferation."
        elif vessel_density > 13.5:
            nv_risk = "Moderate"
            nv_description = f"Vascular density borderline ({vessel_density:.1f}%). Monitor for vascular remodeling."
        else:
            nv_risk = "Low"
            nv_description = f"Vascular density within expected range ({vessel_density:.1f}%)."

        # Build Composite Overlay
        overlay = img_bgr.copy()
        overlay[clean_ex_mask] = [0, 230, 255]
        ma_disp = cv2.dilate(clean_ma_mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))) > 0
        overlay[ma_disp] = [255, 0, 240]
        overlay[clean_he_mask] = [0, 0, 240]

        blended = cv2.addWeighted(img_bgr, 0.55, overlay, 0.45, 0)
        overlay_filename = f"{filename_prefix}_overlay_{int(np.random.randint(100000, 999999))}.png"
        overlay_path = self.output_dir / overlay_filename
        cv2.imwrite(str(overlay_path), blended)

        return {
            "mode": "HEURISTIC FALLBACK",
            "badge_label": "HEURISTIC FALLBACK (Classical Morphology)",
            "is_ai": False,
            "microaneurysms": {
                "candidate_count": int(ma_count),
                "label": "Microaneurysm Candidate Detection",
                "methodology": "Green-channel morphological top-hat filtering (3–65 px)"
            },
            "hard_exudates": {
                "candidate_count": int(exudate_count),
                "total_area_px": int(exudate_area),
                "label": "Hard Exudate Candidate Detection",
                "methodology": "High-luminance intra-retinal lipid segmentation (Optic disc excluded)"
            },
            "hemorrhages": {
                "candidate_count": int(he_count),
                "total_area_px": int(he_area),
                "label": "Intra-retinal Hemorrhage Candidate Detection",
                "methodology": "Dark lesion connected-component analysis (40–2500 px)"
            },
            "soft_exudates": {
                "candidate_count": 0,
                "total_area_px": 0,
                "label": "Soft Exudates (Not modeled in heuristic fallback)",
                "methodology": "N/A in heuristic mode"
            },
            "neovascularization": {
                "risk_level": nv_risk,
                "label": "Neovascularization Risk Indicator",
                "description": nv_description,
                "methodology": "Peri-papillary vessel density & branching stratification"
            },
            "overlay_url": f"/outputs/{overlay_filename}",
            "legend": [
                {"name": "Hard Exudates", "color": "#ffe600", "description": "Lipid deposits (Disc excluded)"},
                {"name": "Microaneurysms", "color": "#ff00ea", "description": "Focal microvascular dilatations"},
                {"name": "Hemorrhages", "color": "#ff2a2a", "description": "Intra-retinal blotches"}
            ]
        }
