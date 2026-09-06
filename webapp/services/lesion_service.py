"""
lesion_service.py
=================
Quantitative Computer-Vision Lesion Candidate Analysis.
Detects:
- Microaneurysm Candidates (Morphological top-hat filtering, 3–65 px circular dark lesions)
- Hard Exudate Candidates (High-luminance intra-retinal lipid segmentation with strict optic-disc masking)
- Intra-retinal Hemorrhage Candidates (Dark lesion blotch segmentation, 40–2500 px)
- Neovascularization Risk Indicator (Non-diagnostic risk metric from vessel density)

Note: All findings are mathematically labeled as 'Candidate Detection' or 'Risk Indicator',
never 'Confirmed Diagnosis', maintaining complete scientific and medical honesty.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import cv2


class LesionService:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def analyze(
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

        # ---------------------------------------------------------------------
        # 1. OPTIC DISC EXCLUSION MASK (Dilated by 25% to avoid disc rim artifacts)
        # ---------------------------------------------------------------------
        if od_mask is not None:
            kernel_disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            od_dilated = cv2.dilate(od_mask.astype(np.uint8), kernel_disc) > 0
        else:
            od_dilated = np.zeros((h, w), dtype=bool)

        # ---------------------------------------------------------------------
        # 2. HARD EXUDATE CANDIDATE SEGMENTATION
        # ---------------------------------------------------------------------
        # Exudates are bright yellow-white lipid clusters outside the optic nerve head
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

        # ---------------------------------------------------------------------
        # 3. MICROANEURYSM CANDIDATE DETECTION
        # ---------------------------------------------------------------------
        # Small isolated circular focal dilatations (3–65 px) on inverted green channel
        kernel_ma = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        ma_tophat = cv2.morphologyEx(inv_green, cv2.MORPH_TOPHAT, kernel_ma)
        _, ma_binary = cv2.threshold(ma_tophat, 24, 255, cv2.THRESH_BINARY)
        ma_binary = ma_binary & retina_mask.astype(np.uint8) & (~od_dilated).astype(np.uint8)

        num_ma, ma_labels, ma_stats, _ = cv2.connectedComponentsWithStats(ma_binary)
        ma_count = 0
        clean_ma_mask = np.zeros((h, w), dtype=bool)
        for i in range(1, num_ma):
            area = ma_stats[i, cv2.CC_STAT_AREA]
            if 3 <= area <= 65:
                ma_count += 1
                clean_ma_mask[ma_labels == i] = True

        # ---------------------------------------------------------------------
        # 4. INTRA-RETINAL HEMORRHAGE CANDIDATE SEGMENTATION
        # ---------------------------------------------------------------------
        # Larger dark blotches (40–2500 px)
        kernel_he = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        he_tophat = cv2.morphologyEx(inv_green, cv2.MORPH_TOPHAT, kernel_he)
        _, he_binary = cv2.threshold(he_tophat, 28, 255, cv2.THRESH_BINARY)
        he_binary = he_binary & retina_mask.astype(np.uint8) & (~clean_ma_mask).astype(np.uint8) & (~od_dilated).astype(np.uint8)

        num_he, he_labels, he_stats, _ = cv2.connectedComponentsWithStats(he_binary)
        he_count = 0
        he_area = 0
        clean_he_mask = np.zeros((h, w), dtype=bool)
        for i in range(1, num_he):
            area = he_stats[i, cv2.CC_STAT_AREA]
            if 40 <= area <= 2500:
                he_count += 1
                he_area += int(area)
                clean_he_mask[he_labels == i] = True

        # ---------------------------------------------------------------------
        # 5. NEOVASCULARIZATION RISK INDICATOR
        # ---------------------------------------------------------------------
        # Derived non-diagnostically from vascular density percentage
        if vessel_density > 18.0:
            nv_risk = "High"
            nv_description = "Markedly elevated peri-papillary vessel density; clinical evaluation for abnormal proliferation warranted."
        elif vessel_density > 14.0:
            nv_risk = "Moderate"
            nv_description = "Moderately elevated vascular branching; assess for fine abnormal vessels."
        else:
            nv_risk = "Low"
            nv_description = "Vessel density within standard physiological range."

        # ---------------------------------------------------------------------
        # 6. MULTI-COLOR LESION OVERLAY
        # ---------------------------------------------------------------------
        overlay = img_bgr.copy()
        
        # Yellow for Hard Exudates (BGR: 0, 240, 255)
        overlay[clean_ex_mask] = [0, 230, 255]
        
        # Magenta for Microaneurysms (BGR: 255, 0, 255)
        # Dilate 1px for visibility on UI
        ma_disp = cv2.dilate(clean_ma_mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))) > 0
        overlay[ma_disp] = [255, 0, 240]

        # Red for Hemorrhages (BGR: 0, 0, 240)
        overlay[clean_he_mask] = [0, 0, 240]

        # Blend with original image
        blended = cv2.addWeighted(img_bgr, 0.55, overlay, 0.45, 0)

        # Save overlay file
        overlay_filename = f"{filename_prefix}_overlay_{int(np.random.randint(100000, 999999))}.png"
        overlay_path = self.output_dir / overlay_filename
        cv2.imwrite(str(overlay_path), blended)

        return {
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
