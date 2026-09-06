"""
quality_service.py
==================
Real Image Quality Assessment for RetinaGuard Web Workstation.
Computes objective physical metrics from the retinal fundus image:
- Sharpness (Laplacian gradient variance)
- Illumination (Mean green-channel luminance)
- Contrast (Standard deviation & dynamic range)
- Field of View (FOV percentage of active retinal area)
- Overall Quality Score (0–100) & Clinical Status (GOOD, BORDERLINE, UNGRADABLE)
"""

from typing import Dict, Any
import numpy as np
import cv2


class QualityService:
    def __init__(self, min_score: float = 40.0, borderline_score: float = 60.0):
        self.min_score = min_score
        self.borderline_score = borderline_score

    def assess(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Assesses clinical quality of a retinal fundus image.
        
        Args:
            image_bgr: numpy array (H, W, 3) in BGR format
            
        Returns:
            Dictionary with quality scores, sub-metrics, status, and recapture advice.
        """
        if image_bgr is None or image_bgr.size == 0:
            return {
                "score": 0.0,
                "status": "UNGRADABLE",
                "gradable": False,
                "focus": "Poor",
                "illumination": "Poor",
                "contrast": "Poor",
                "field_of_view": "Poor",
                "snr_db": 0.0,
                "reason": "Empty or corrupted image data.",
                "recommendation": "Recapture retinal image with valid camera output."
            }

        h, w = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        green = image_bgr[:, :, 1]

        # 1. Sharpness / Focus via Laplacian Variance
        lap_var = float(cv2.Laplacian(green, cv2.CV_64F).var())
        # Map lap_var (typically 0-500+) to 0-100 score
        sharpness_score = min(100.0, (lap_var / 450.0) * 100.0)
        focus_label = "Good" if lap_var >= 100 else ("Acceptable" if lap_var >= 45 else "Poor")

        # 2. Illumination (Mean intensity of green channel)
        mean_lum = float(np.mean(green))
        # Ideal retinal luminance center is around 120-130
        illum_dev = abs(mean_lum - 125.0)
        illum_score = max(0.0, 100.0 - (illum_dev * (100.0 / 115.0)))
        illum_label = "Good" if (60 <= mean_lum <= 180) else ("Acceptable" if (35 <= mean_lum <= 225) else "Poor")

        # 3. Contrast (Standard deviation of active pixels)
        active_mask = green > 15
        if np.sum(active_mask) > 100:
            contrast_std = float(np.std(green[active_mask]))
        else:
            contrast_std = float(np.std(green))
        contrast_score = min(100.0, (contrast_std / 55.0) * 100.0)
        contrast_label = "Good" if contrast_std >= 35 else ("Acceptable" if contrast_std >= 20 else "Poor")

        # 4. Field of View (FOV percentage)
        total_pixels = h * w
        fov_percent = (float(np.sum(active_mask)) / float(total_pixels)) * 100.0
        fov_score = min(100.0, (fov_percent / 65.0) * 100.0)
        fov_label = "Good" if fov_percent >= 50.0 else ("Acceptable" if fov_percent >= 30.0 else "Poor")

        # 5. Signal-to-Noise Ratio (SNR) proxy
        noise_est = float(np.median(np.abs(lap_var - np.mean(lap_var))))
        snr_db = round(20.0 * np.log10(max(1.0, mean_lum) / max(1.0, noise_est + 1e-4)), 1)

        # Overall Composite Score (0–100)
        overall_score = (
            0.45 * sharpness_score +
            0.25 * illum_score +
            0.20 * contrast_score +
            0.10 * fov_score
        )
        overall_score = round(float(np.clip(overall_score, 0.0, 100.0)), 1)

        # Classification decision rules
        reasons = []
        if lap_var < 35.0:
            reasons.append("Image is severely blurred / out of focus")
        if mean_lum < 30.0:
            reasons.append("Severe underexposure / darkness")
        elif mean_lum > 225.0:
            reasons.append("Severe overexposure / sensor glare")
        if fov_percent < 25.0:
            reasons.append("Insufficient retinal field of view (< 25% of sensor area)")

        if overall_score < self.min_score or len(reasons) > 0:
            status = "UNGRADABLE"
            gradable = False
            primary_reason = "; ".join(reasons) if reasons else "Composite image quality is below clinical diagnostic threshold."
            rec = "Recapture retinal image with improved focus, centered illumination, and full pupil dilation."
        elif overall_score < self.borderline_score:
            status = "BORDERLINE"
            gradable = True
            primary_reason = "Borderline focus or contrast; subtle micro-lesions may be difficult to discern."
            rec = "Automated enhancement applied. Proceed with caution; human ophthalmologist review recommended."
        else:
            status = "GOOD"
            gradable = True
            primary_reason = "Retinal image meets all clinical quality criteria for automated screening."
            rec = "Proceed with automated screening pipeline."

        return {
            "score": overall_score,
            "status": status,
            "gradable": gradable,
            "focus": focus_label,
            "illumination": illum_label,
            "contrast": contrast_label,
            "field_of_view": fov_label,
            "snr_db": snr_db,
            "laplacian_var": round(lap_var, 1),
            "mean_lum": round(mean_lum, 1),
            "fov_percent": round(fov_percent, 1),
            "reason": primary_reason,
            "recommendation": rec
        }
