"""
segment_lesions.py
==================
Production Inference Engine for Retinal Lesion and Anatomical Segmentation.

Features:
- Primary Engine: Real-time Dual-Head U-Net ONNX model (models/experiments/lesion_segmentation/lesion_unet.onnx)
- Graceful Fallback: Mathematical morphology & filter heuristic detector if model is absent
- Honest Reporting: Explicitly flags mode as 'AI SEGMENTATION' or 'HEURISTIC FALLBACK'
- Decouples Lesion Pathology (MA, HE, EX, SE) from Normal Anatomy (Optic Disc)
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = ROOT / "models" / "experiments" / "lesion_segmentation"
ONNX_PATH = MODEL_DIR / "lesion_unet.onnx"
PT_PATH = MODEL_DIR / "best_model.pt"

# Color definitions for visualization (RGB)
COLORS = {
    "MA": (255, 23, 68),     # Vibrant Red for Microaneurysms
    "HE": (255, 145, 0),     # Amber/Orange for Haemorrhages
    "EX": (255, 234, 0),     # Bright Yellow for Hard Exudates
    "SE": (0, 229, 255),     # Electric Cyan for Soft Exudates
    "OD": (0, 230, 118),     # Emerald Green for Optic Disc (Normal Anatomy)
}


class LesionSegmentationPipeline:
    def __init__(self, model_path: Optional[Path] = None, threshold: float = 0.5):
        self.model_path = model_path or ONNX_PATH
        self.threshold = threshold
        self.onnx_session = None
        self.pt_model = None
        self.device = "cpu"
        self.is_ai_loaded = False

        self._initialize_model()

    def _initialize_model(self):
        # 1. Try loading ONNX Runtime session
        if self.model_path.exists():
            try:
                import onnxruntime as ort
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 2
                self.onnx_session = ort.InferenceSession(str(self.model_path), sess_options=opts, providers=["CPUExecutionProvider"])
                self.is_ai_loaded = True
                print(f"[LesionSegmentation] Initialized ONNX inference session: {self.model_path.name}")
                return
            except Exception as e:
                print(f"[LesionSegmentation] ONNX init failed ({e}), attempting PyTorch fallback...")

        # 2. Try loading PyTorch checkpoint
        if PT_PATH.exists():
            try:
                import torch
                from python.segmentation.unet import DualHeadUNet
                self.pt_model = DualHeadUNet(export_mode=True)
                ckpt = torch.load(PT_PATH, map_location="cpu")
                state = ckpt.get("model_state_dict", ckpt)
                self.pt_model.load_state_dict(state)
                self.pt_model.eval()
                self.is_ai_loaded = True
                print(f"[LesionSegmentation] Initialized PyTorch inference model: {PT_PATH.name}")
                return
            except Exception as e:
                print(f"[LesionSegmentation] PyTorch init failed ({e}).")

        print("[LesionSegmentation] No AI model available. System will run in HEURISTIC FALLBACK mode.")
        self.is_ai_loaded = False

    def segment(self, image_rgb: np.ndarray) -> Dict[str, Any]:
        """
        Segments lesions and optic disc from an RGB retinal fundus photograph.

        Args:
            image_rgb: numpy uint8 array [H, W, 3]

        Returns:
            Structured dictionary with masks, areas, counts, and active mode.
        """
        if self.is_ai_loaded:
            try:
                return self._segment_ai(image_rgb)
            except Exception as e:
                print(f"[LesionSegmentation] AI inference exception ({e}). Falling back to heuristic...")
                return self._segment_heuristic(image_rgb)
        else:
            return self._segment_heuristic(image_rgb)

    def _preprocess(self, image_rgb: np.ndarray, target_size: int = 256) -> np.ndarray:
        resized = cv2.resize(image_rgb, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        float_img = resized.astype(np.float32) / 255.0
        # Normalize with ImageNet mean/std
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm_img = (float_img - mean) / std
        # [H, W, C] -> [1, C, H, W]
        transposed = np.transpose(norm_img, (2, 0, 1))
        return np.expand_dims(transposed, axis=0).astype(np.float32)

    def _segment_ai(self, image_rgb: np.ndarray) -> Dict[str, Any]:
        h_orig, w_orig = image_rgb.shape[:2]
        input_tensor = self._preprocess(image_rgb, target_size=256)

        # Run forward pass
        if self.onnx_session is not None:
            input_name = self.onnx_session.get_inputs()[0].name
            outputs = self.onnx_session.run(None, {input_name: input_tensor})
            sigmoid_probs = outputs[0][0]  # [5, 256, 256]
        else:
            import torch
            with torch.no_grad():
                out = self.pt_model(torch.from_numpy(input_tensor))
                sigmoid_probs = out.numpy()[0]  # [5, 256, 256]

        # Channels: 0: MA, 1: HE, 2: EX, 3: SE, 4: OD
        calibrated_thresholds = {
            "MA": 0.15,
            "HE": 0.15,
            "EX": 0.20,
            "SE": 0.30,
            "OD": 0.50
        }

        lesion_keys = ["MA", "HE", "EX", "SE"]
        lesion_names = {
            "MA": "Microaneurysms",
            "HE": "Haemorrhages",
            "EX": "Hard Exudates",
            "SE": "Soft Exudates"
        }

        lesion_masks_resized = {}
        lesion_counts = {}
        lesion_areas = {}
        lesion_pcts = {}

        total_pixels = h_orig * w_orig

        # Process the 4 lesions
        for c, key in enumerate(lesion_keys):
            thresh = calibrated_thresholds[key]
            mask_256 = (sigmoid_probs[c] > thresh).astype(np.uint8)
            mask_orig = cv2.resize(mask_256, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
            lesion_masks_resized[key] = mask_orig

            num_labels, _, _, _ = cv2.connectedComponentsWithStats(mask_orig)
            count = max(0, num_labels - 1)  # subtract background
            area = int(np.sum(mask_orig > 0))
            pct = round((area / total_pixels) * 100, 4) if total_pixels > 0 else 0.0

            lesion_counts[key] = count
            lesion_areas[key] = area
            lesion_pcts[key] = pct

        # Process Optic Disc (Normal Anatomy, thresh=0.50)
        od_mask_256 = (sigmoid_probs[4] > calibrated_thresholds["OD"]).astype(np.uint8)
        od_mask_orig = cv2.resize(od_mask_256, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        od_area = int(np.sum(od_mask_orig > 0))
        od_pct = round((od_area / total_pixels) * 100, 4) if total_pixels > 0 else 0.0

        # Calculate OD centroid if present
        od_centroid = None
        if od_area > 0:
            m = cv2.moments(od_mask_orig)
            if m["m00"] > 0:
                od_centroid = (int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"]))

        # Build colored overlay
        overlay = self._create_overlay(image_rgb, lesion_masks_resized, od_mask_orig)

        return {
            "mode": "AI SEGMENTATION",
            "is_ai": True,
            "badge_label": "AI SEGMENTATION (Dual-Head U-Net)",
            "lesion_description": "AI-detected lesion regions (trained on expert annotations)",
            "lesion_counts": lesion_counts,
            "lesion_areas": lesion_areas,
            "lesion_pcts": lesion_pcts,
            "lesion_masks": lesion_masks_resized,
            "anatomy": {
                "optic_disc_mask": od_mask_orig,
                "optic_disc_area": od_area,
                "optic_disc_pct": od_pct,
                "optic_disc_centroid": od_centroid,
                "category": "Normal Anatomy"
            },
            "overlay_rgb": overlay
        }

    def _segment_heuristic(self, image_rgb: np.ndarray) -> Dict[str, Any]:
        """Heuristic fallback using green channel contrast and morphological filters."""
        h_orig, w_orig = image_rgb.shape[:2]
        green = image_rgb[:, :, 1]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        g_enh = clahe.apply(green)
        total_pixels = h_orig * w_orig

        # 1. Microaneurysm candidates (black top-hat morphology)
        kernel_ma = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        tophat_ma = cv2.morphologyEx(g_enh, cv2.MORPH_BLACKHAT, kernel_ma)
        _, ma_mask = cv2.threshold(tophat_ma, 25, 1, cv2.THRESH_BINARY)
        # 2. Hard exudate candidates (bright lesions)
        tophat_ex = cv2.morphologyEx(g_enh, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        _, ex_mask = cv2.threshold(tophat_ex, 35, 1, cv2.THRESH_BINARY)
        # 3. Hemorrhages (larger dark regions)
        tophat_he = cv2.morphologyEx(g_enh, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
        _, he_mask = cv2.threshold(tophat_he, 30, 1, cv2.THRESH_BINARY)
        # 4. Soft exudates (cotton wool spots)
        se_mask = np.zeros_like(ma_mask)

        lesion_masks = {"MA": ma_mask, "HE": he_mask, "EX": ex_mask, "SE": se_mask}
        lesion_counts = {}
        lesion_areas = {}
        lesion_pcts = {}

        for k, mask in lesion_masks.items():
            num_labels, _, _, _ = cv2.connectedComponentsWithStats(mask)
            count = max(0, num_labels - 1)
            area = int(np.sum(mask > 0))
            lesion_counts[k] = count
            lesion_areas[k] = area
            lesion_pcts[k] = round((area / total_pixels) * 100, 4) if total_pixels > 0 else 0.0

        # Heuristic Optic Disc
        od_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(g_enh)
        cv2.circle(od_mask, max_loc, int(min(h_orig, w_orig) * 0.08), 1, -1)
        od_area = int(np.sum(od_mask > 0))

        overlay = self._create_overlay(image_rgb, lesion_masks, od_mask)

        return {
            "mode": "HEURISTIC FALLBACK",
            "is_ai": False,
            "badge_label": "HEURISTIC FALLBACK (Classical Morphology)",
            "lesion_description": "Candidate lesion detections (morphological edge & filter heuristics)",
            "lesion_counts": lesion_counts,
            "lesion_areas": lesion_areas,
            "lesion_pcts": lesion_pcts,
            "lesion_masks": lesion_masks,
            "anatomy": {
                "optic_disc_mask": od_mask,
                "optic_disc_area": od_area,
                "optic_disc_pct": round((od_area / total_pixels) * 100, 4),
                "optic_disc_centroid": max_loc,
                "category": "Normal Anatomy"
            },
            "overlay_rgb": overlay
        }

    def _create_overlay(self, image_rgb: np.ndarray, lesion_masks: Dict[str, np.ndarray], od_mask: np.ndarray) -> np.ndarray:
        overlay = image_rgb.copy()
        color_mask = np.zeros_like(image_rgb)

        # Draw lesions
        for key in ["SE", "EX", "HE", "MA"]:  # smaller lesions layered on top
            mask = lesion_masks.get(key)
            if mask is not None and np.sum(mask) > 0:
                color = COLORS[key]
                color_mask[mask > 0] = color

        # Draw Optic Disc border
        if od_mask is not None and np.sum(od_mask) > 0:
            contours, _ = cv2.findContours(od_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(color_mask, contours, -1, COLORS["OD"], 3)

        if np.sum(color_mask) > 0:
            overlay = cv2.addWeighted(overlay, 0.65, color_mask, 0.35, 0)

        return overlay
