"""
RetinaGuard — Inference API
============================
Clean predictDR() function for single-image DR severity prediction.

Usage (from Python):
    from inference.predict import DRPredictor
    predictor = DRPredictor(cfg)
    result = predictor.predict(pil_image)

Returns:
    {
        "grade": 2,
        "severity": "Moderate DR",
        "confidence": 0.83,
        "confidence_level": "high",   # "high" | "moderate" | "uncertain"
        "probabilities": [0.05, 0.08, 0.83, 0.03, 0.01],
        "referral": True,
        "referral_text": "Ophthalmology evaluation recommended within 3-6 months.",
        "demo_mode": False,
        "model_info": "APTOS EfficientNet-B0"
    }

The inference pipeline MUST use the same preprocessing as training.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

CLASS_NAMES = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]
REFERRAL_THRESHOLD = 2  # Grade >= 2 → recommend ophthalmologist evaluation

REFERRAL_TEXTS = {
    0: "No signs of diabetic retinopathy detected. Routine follow-up in 12 months. Maintain glycemic control.",
    1: "Mild signs detected. Consider follow-up in 6–12 months. Optimise glycemic and blood pressure control.",
    2: "Moderate non-proliferative DR. Ophthalmology evaluation recommended within 3–6 months.",
    3: "Severe non-proliferative DR. Ophthalmology referral required within 1 month.",
    4: "Proliferative DR detected. Urgent ophthalmology referral required. Vision-threatening condition.",
}


class DRPredictor:
    """
    Production-ready inference class for DR severity prediction.

    Loads the trained model once at construction.
    Subsequent predict() calls are fast (no model reloading).

    Falls back to demo mode gracefully if no model file is found.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._root = Path(cfg.get("_root", "."))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.demo_mode = True
        self.model_info = "Demo Mode (no model loaded)"

        self._uncertainty_threshold = float(
            cfg.get("inference", {}).get("uncertainty_threshold", 0.65)
        )

        # Try to load the trained model
        self._try_load_model()

        # Build preprocessor (same pipeline as training)
        import sys
        sys.path.insert(0, str(self._root / "python"))
        from preprocessing.retinal_preprocessor import RetinalPreprocessor
        self.preprocessor = RetinalPreprocessor(cfg)

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def predict(self, image, generate_gradcam: bool = False) -> Dict:
        """
        Predict DR severity for a single retinal image.

        Args:
            image: PIL.Image or numpy array (H×W×3 uint8)
            generate_gradcam: Whether to generate a Grad-CAM heatmap overlay

        Returns:
            result dict (see module docstring)
        """
        from PIL import Image as PILImage

        # Normalise input to PIL
        if not isinstance(image, PILImage.Image):
            image = PILImage.fromarray(np.uint8(image))
        if image.mode != "RGB":
            image = image.convert("RGB")

        if self.demo_mode or self.model is None:
            return self._demo_predict(image)

        return self._real_predict(image, generate_gradcam=generate_gradcam)

    def is_loaded(self) -> bool:
        return not self.demo_mode

    def get_status(self) -> str:
        return self.model_info

    # ──────────────────────────────────────────────────────────────────────────
    # Real model inference
    # ──────────────────────────────────────────────────────────────────────────

    def _real_predict(self, image, generate_gradcam: bool = False) -> Dict:
        """Run inference using the trained EfficientNet-B0 model."""
        try:
            # Preprocess
            tensor = self.preprocessor.preprocess_for_inference(image)
            tensor = tensor.to(self.device)

            # Forward pass
            self.model.eval()
            with torch.no_grad():
                logits = self.model(tensor)
                probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()

            grade = int(np.argmax(probs))
            confidence = float(probs[grade])
            confidence_level = self._confidence_level(confidence)

            gradcam_path = None
            if generate_gradcam:
                try:
                    import tempfile
                    from explainability.gradcam import GradCAM
                    gcam = GradCAM(self.model)
                    gcam_tensor = self.preprocessor.preprocess_for_inference(image)
                    heatmap_np, _ = gcam.generate(gcam_tensor, target_class=grade)
                    overlay = gcam.overlay_on_image(image, heatmap_np, alpha=0.45)
                    gcam.remove_hooks()
                    temp_dir = Path(tempfile.gettempdir())
                    out_gcam = temp_dir / "retinaguard_gradcam_active.png"
                    overlay.save(out_gcam)
                    gradcam_path = str(out_gcam)
                except Exception as ex:
                    logger.warning("Grad-CAM generation failed: %s", ex)

            return {
                "grade": grade,
                "severity": CLASS_NAMES[grade],
                "confidence": round(confidence, 4),
                "confidence_level": confidence_level,
                "probabilities": [round(float(p), 4) for p in probs],
                "referral": grade >= REFERRAL_THRESHOLD,
                "referral_text": REFERRAL_TEXTS[grade],
                "demo_mode": False,
                "model_info": self.model_info,
                "gradcam_path": gradcam_path,
                "gradcam_available": gradcam_path is not None,
            }

        except Exception as e:
            logger.error("Inference failed: %s", e)
            return self._error_result(str(e))

    # ──────────────────────────────────────────────────────────────────────────
    # Demo mode fallback
    # ──────────────────────────────────────────────────────────────────────────

    def _demo_predict(self, image) -> Dict:
        """
        Demo mode: returns a plausible but clearly labelled fake prediction.
        Used only when no trained model is available.
        """
        # Use image statistics as a rough (non-clinical) heuristic
        import numpy as np
        arr = np.array(image).astype(float)
        mean_brightness = arr.mean() / 255.0
        # Very rough grade proxy — for demo display only, clearly labelled
        grade = min(4, max(0, int((1.0 - mean_brightness) * 5)))
        probs = np.zeros(5)
        probs[grade] = 0.72
        for i in range(5):
            if i != grade:
                probs[i] = 0.28 / 4
        probs = probs / probs.sum()

        return {
            "grade": grade,
            "severity": CLASS_NAMES[grade],
            "confidence": round(float(probs[grade]), 4),
            "confidence_level": "uncertain",
            "probabilities": [round(float(p), 4) for p in probs],
            "referral": grade >= REFERRAL_THRESHOLD,
            "referral_text": REFERRAL_TEXTS[grade],
            "demo_mode": True,
            "model_info": "DEMO MODE — No trained model loaded",
        }

    def _error_result(self, error_msg: str) -> Dict:
        return {
            "grade": -1,
            "severity": "Error",
            "confidence": 0.0,
            "confidence_level": "uncertain",
            "probabilities": [0.2, 0.2, 0.2, 0.2, 0.2],
            "referral": False,
            "referral_text": f"Inference error: {error_msg}",
            "demo_mode": True,
            "model_info": f"Error: {error_msg}",
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Model loading
    # ──────────────────────────────────────────────────────────────────────────

    def _try_load_model(self):
        model_path = self._root / self.cfg["paths"]["best_model"]
        if not model_path.exists():
            logger.info("No trained model at %s — running in Demo Mode.", model_path)
            return

        try:
            import sys
            sys.path.insert(0, str(self._root / "python"))
            from models.efficientnet_dr import build_model

            self.model = build_model(self.cfg)
            state = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state)
            self.model = self.model.to(self.device)
            self.model.eval()

            # Load metadata if available
            meta_path = model_path.parent / "metadata.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                arch = meta.get("architecture", "efficientnet_b0")
                dataset = meta.get("dataset", "APTOS")
                date = meta.get("training_date", "")[:10]
                self.model_info = f"APTOS {arch} (trained {date})"
            else:
                self.model_info = "APTOS EfficientNet-B0"

            self.demo_mode = False
            logger.info("Model loaded: %s", self.model_info)

        except Exception as e:
            logger.error("Failed to load model (%s): %s", model_path, e)
            self.model = None
            self.demo_mode = True

    def _confidence_level(self, confidence: float) -> str:
        if confidence >= 0.80:
            return "high"
        elif confidence >= self._uncertainty_threshold:
            return "moderate"
        else:
            return "uncertain"


# ==============================================================================
#  Convenience function (matches spec: predictDR(image))
# ==============================================================================

_default_predictor: Optional[DRPredictor] = None


def predictDR(image, cfg: Optional[dict] = None, generate_gradcam: bool = False) -> Dict:
    """
    Module-level convenience function.
    First call loads the model; subsequent calls reuse it.

    Usage:
        from inference.predict import predictDR
        result = predictDR(pil_image)
    """
    global _default_predictor

    if _default_predictor is None:
        if cfg is None:
            import sys, yaml
            python_dir = Path(__file__).parent.parent
            config_path = python_dir / "config" / "config.yaml"
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            cfg["_root"] = str(python_dir.parent)
        _default_predictor = DRPredictor(cfg)

    return _default_predictor.predict(image, generate_gradcam=generate_gradcam)


# ==============================================================================
#  CLI test
# ==============================================================================

if __name__ == "__main__":
    import argparse, sys, yaml
    from PIL import Image

    parser = argparse.ArgumentParser(description="Test inference on a single image")
    parser.add_argument("image", nargs="?", help="Path to retinal image")
    parser.add_argument("--config", default=None)
    parser.add_argument("--json", action="store_true", help="Output pure JSON (for MATLAB bridge)")
    parser.add_argument("--gradcam", action="store_true", help="Generate Grad-CAM overlay")
    args = parser.parse_args()

    python_dir = Path(__file__).parent.parent
    config_path = args.config or str(python_dir / "config" / "config.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = str(python_dir.parent)

    predictor = DRPredictor(cfg)

    if args.json:
        if args.image:
            img = Image.open(args.image).convert("RGB")
            result = predictor.predict(img, generate_gradcam=args.gradcam)
            print(json.dumps(result))
        else:
            print(json.dumps({"error": "No image path provided"}))
        sys.exit(0)

    print(f"\nModel status: {predictor.get_status()}")
    print(f"Demo mode: {predictor.demo_mode}\n")

    if args.image:
        img = Image.open(args.image).convert("RGB")
        result = predictor.predict(img, generate_gradcam=args.gradcam)
        print("Inference result:")
        for k, v in result.items():
            print(f"  {k:<22}: {v}")
    else:
        print("Usage: python inference/predict.py <image_path>")
        print("No image provided -- predictor initialised OK.")
