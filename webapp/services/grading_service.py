"""
grading_service.py
==================
Deep Learning DR Severity Grading & Grad-CAM Explainability Service.
Uses the locked production model: EXP-001 (EfficientNet-B0).
Outputs:
- 5-class ICDR severity classification (Grades 0 to 4)
- Exact class probabilities from softmax output
- Referable DR classification (Grade >= 2)
- True Grad-CAM saliency heatmap overlay highlighting pathology
- Actionable clinical screening recommendations
"""

import sys
from pathlib import Path
from typing import Dict, Any
import numpy as np
from PIL import Image
import yaml
import torch
import torch.nn.functional as F

# Ensure python directory is importable
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "python"))

from inference.predict import DRPredictor, CLASS_NAMES, REFERRAL_TEXTS, REFERRAL_THRESHOLD
from explainability.gradcam import GradCAM


class GradingService:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load project configuration
        config_path = ROOT / "python" / "config" / "config.yaml"
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)
        self.cfg["_root"] = str(ROOT)

        # Initialize DRPredictor using production model
        self.predictor = DRPredictor(self.cfg)
        self.model_info = "EXP-001 (EfficientNet-B0)"

        # Load production ONNX model via config-based relative path
        onnx_rel_path = self.cfg.get("paths", {}).get("onnx_model", "models/aptos_efficientnet/best_model.onnx")
        self.onnx_path = ROOT / onnx_rel_path
        self.onnx_session = None
        self.onnx_input_name = None
        if self.onnx_path.exists():
            try:
                import onnxruntime as ort
                self.onnx_session = ort.InferenceSession(
                    str(self.onnx_path),
                    providers=["CPUExecutionProvider"]
                )
                self.onnx_input_name = self.onnx_session.get_inputs()[0].name
            except Exception:
                self.onnx_session = None

    def grade(self, pil_image: Image.Image, filename_prefix: str = "gradcam") -> Dict[str, Any]:
        """
        Runs deep learning inference and generates Grad-CAM heatmap.
        Uses production ONNX model (EXP-001) with PyTorch fallback.
        """
        import time
        # 1. Forward pass (Primary: ONNX runtime, Fallback: PyTorch)
        print("[SCREENING] ONNX Inference START")
        t_onnx_0 = time.time()
        tensor = self.predictor.preprocessor.preprocess_for_inference(pil_image)
        if self.onnx_session is not None:
            ort_inputs = {self.onnx_input_name: tensor.cpu().numpy()}
            logits = self.onnx_session.run(None, ort_inputs)[0]
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = (exp_logits / np.sum(exp_logits, axis=1, keepdims=True)).squeeze()
        else:
            self.predictor.model.eval()
            with torch.no_grad():
                logits = self.predictor.model(tensor.to(self.predictor.device))
                probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
        t_onnx_1 = time.time()
        print(f"[SCREENING] ONNX Inference COMPLETE {t_onnx_1 - t_onnx_0:.2f}s")

        grade = int(np.argmax(probs))
        confidence = float(probs[grade])
        is_referable = bool(grade >= REFERRAL_THRESHOLD)
        conf_level = "High" if confidence >= 0.80 else ("Moderate" if confidence >= 0.50 else "Borderline")

        # Build clean probabilities mapping
        class_probs = []
        labels = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"]
        for idx, p in enumerate(probs):
            class_probs.append({
                "grade": idx,
                "label": labels[idx],
                "probability": round(float(p), 4),
                "percentage": round(float(p) * 100.0, 1)
            })

        return {
            "grade": grade,
            "severity_name": labels[grade],
            "confidence": round(confidence, 4),
            "confidence_percent": round(confidence * 100.0, 1),
            "confidence_level": conf_level,
            "probabilities": class_probs,
            "referral": is_referable,
            "referral_badge": "REFERABLE DR" if is_referable else "NON-REFERABLE",
            "referral_action": "Ophthalmology Referral Required" if is_referable else "Routine Follow-up Advised",
            "recommendation": REFERRAL_TEXTS[grade],
            "model_version": "EXP-001",
            "architecture": "EfficientNet-B0 (Focal Loss, Balanced Sampling)",
            "gradcam": {
                "available": False,
                "status": "ready_for_generation",
                "endpoint": "/api/gradcam",
                "url": None,
                "error": None
            },
            "gradcam_url": None,
            "gradcam_available": False,
            "gradcam_error": None,
            "gradcam_disclaimer": "Model Attention Map: Highlights spatial regions that contributed most strongly to the neural network prediction. This is an explainability tool, not a manual lesion segmentation."
        }

    def generate_gradcam(
        self,
        pil_image: Image.Image,
        target_grade: int,
        filename_prefix: str = "gradcam",
        timeout_sec: float = 25.0
    ) -> Dict[str, Any]:
        """
        Executes REAL PyTorch Grad-CAM with hard server-side execution timeout.
        Never blocks server indefinitely; never affects core screening.
        """
        import time
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

        t_gcam_0 = time.time()
        print(f"[SCREENING] Dedicated Grad-CAM START (Timeout budget: {timeout_sec}s)")

        def _compute():
            gcam = None
            try:
                for p in self.predictor.model.parameters():
                    p.requires_grad = False

                gcam = GradCAM(self.predictor.model)
                gcam_tensor = self.predictor.preprocessor.preprocess_for_inference(pil_image)
                heatmap, _ = gcam.generate(gcam_tensor, target_class=target_grade)
                overlay = gcam.overlay_on_image(pil_image, heatmap, alpha=0.48, colormap="jet")

                gcam_filename = f"{filename_prefix}_{int(time.time() * 1000) % 1000000}.png"
                gcam_path = self.output_dir / gcam_filename
                overlay.save(str(gcam_path))

                return {
                    "gradcam_available": True,
                    "gradcam_url": f"/outputs/{gcam_filename}",
                    "error": None,
                    "filename": gcam_filename,
                    "disclaimer": "Model Attention Map: Highlights spatial regions that contributed most strongly to the neural network prediction. This is an explainability tool, not a manual lesion segmentation."
                }
            finally:
                if gcam is not None:
                    try:
                        gcam.remove_hooks()
                    except Exception:
                        pass

        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(_compute)
        try:
            res = future.result(timeout=timeout_sec)
            executor.shutdown(wait=False)
            t_elapsed = time.time() - t_gcam_0
            print(f"[SCREENING] Grad-CAM COMPLETE {t_elapsed:.2f}s")
            return res
        except (TimeoutError, FuturesTimeout):
            executor.shutdown(wait=False)
            t_elapsed = time.time() - t_gcam_0
            print(f"[SCREENING] Grad-CAM TIMEOUT exceeded {timeout_sec}s CPU budget ({t_elapsed:.2f}s)")
            return {
                "gradcam_available": False,
                "error": f"Grad-CAM explainability exceeded the available CPU budget ({timeout_sec}s).",
                "gradcam_url": None,
                "disclaimer": "Explainability map temporarily unavailable on this server."
            }
        except Exception as e:
            executor.shutdown(wait=False)
            t_elapsed = time.time() - t_gcam_0
            print(f"[SCREENING] Grad-CAM EXCEPTION ({t_elapsed:.2f}s): {e}")
            return {
                "gradcam_available": False,
                "error": f"Grad-CAM explainability unavailable: {str(e)}",
                "gradcam_url": None,
                "disclaimer": "Explainability map temporarily unavailable on this server."
            }
