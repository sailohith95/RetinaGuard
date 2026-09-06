"""
RetinaGuard — Quick Pipeline Verification Script
==================================================
Tests the complete pipeline with a synthetic image (no dataset needed).
Verifies: preprocessing, model construction, inference API, explainability.

Usage:
    python python/verify_pipeline.py
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PYTHON_DIR = Path(__file__).parent
PROJECT_ROOT = PYTHON_DIR.parent
sys.path.insert(0, str(PYTHON_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("retinaguard.verify")

import yaml


def make_synthetic_fundus(size=224):
    """Create a synthetic retinal-like image for testing."""
    from PIL import Image, ImageDraw
    import math

    img = Image.new("RGB", (size, size), (30, 15, 10))
    draw = ImageDraw.Draw(img)
    cx, cy, r = size // 2, size // 2, size // 2 - 4

    # Retinal disc
    for ri in range(r, 0, -1):
        brightness = int(60 * ri / r + 20)
        redness = int(140 * ri / r + 40)
        draw.ellipse([cx-ri, cy-ri, cx+ri, cy+ri], fill=(redness, brightness//2, brightness//4))

    # Optic disc
    od_x, od_y = int(cx + r * 0.33), cy
    draw.ellipse([od_x-15, od_y-12, od_x+15, od_y+12], fill=(220, 190, 120))

    # Vessels
    for i in range(12):
        angle = i * (2 * math.pi / 12)
        x2 = int(cx + r * 0.85 * math.cos(angle))
        y2 = int(cy + r * 0.85 * math.sin(angle))
        draw.line([cx, cy, x2, y2], fill=(0, 0, 0), width=1)

    return img


def main():
    config_path = PYTHON_DIR / "config" / "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = str(PROJECT_ROOT)

    passed = 0
    failed = 0

    print("\n" + "=" * 65)
    print("  RetinaGuard Python Pipeline Verification")
    print("=" * 65)

    # ── Test 1: Config ──────────────────────────────────────────────────
    try:
        assert cfg["dataset"]["num_classes"] == 5
        assert cfg["model"]["architecture"] == "efficientnet_b0"
        print("[PASS] Config loaded correctly")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Config: {e}")
        failed += 1

    # ── Test 2: Synthetic image ─────────────────────────────────────────
    try:
        img = make_synthetic_fundus(224)
        assert img.size == (224, 224) and img.mode == "RGB"
        print(f"[PASS] Synthetic fundus: {img.size}, mode={img.mode}")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Synthetic fundus: {e}")
        failed += 1
        img = Image.new("RGB", (224, 224))

    # ── Test 3: Preprocessor ────────────────────────────────────────────
    try:
        from preprocessing.retinal_preprocessor import RetinalPreprocessor
        prep = RetinalPreprocessor(cfg)
        tensor_val = prep.transform_val(img)
        tensor_train = prep.transform_train(img)
        assert tensor_val.shape == (3, 224, 224)
        assert tensor_train.shape == (3, 224, 224)
        print(f"[PASS] Preprocessor: val tensor {tuple(tensor_val.shape)}, "
              f"train tensor {tuple(tensor_train.shape)}")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Preprocessor: {e}")
        failed += 1

    # ── Test 4: Model construction ──────────────────────────────────────
    try:
        from models.efficientnet_dr import build_model
        model = build_model(cfg)
        params = model.get_trainable_params()
        print(f"[PASS] Model built: {params['total']:,} params "
              f"({params['trainable']:,} trainable)")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Model construction: {e}")
        failed += 1
        model = None

    # ── Test 5: Model forward pass ──────────────────────────────────────
    if model is not None:
        try:
            model.eval()
            with torch.no_grad():
                dummy = torch.randn(1, 3, 224, 224)
                logits = model(dummy)
            assert logits.shape == (1, 5), f"Expected (1,5) got {logits.shape}"
            probs = torch.softmax(logits, dim=1)
            assert abs(probs.sum().item() - 1.0) < 1e-5
            print(f"[PASS] Forward pass: logits {tuple(logits.shape)}, probs sum ~= 1.0")
            passed += 1
        except Exception as e:
            print(f"[FAIL] Forward pass: {e}")
            failed += 1

    # -- Test 6: Freeze/unfreeze ─────────────────────────────────────────
    if model is not None:
        try:
            model.freeze_backbone()
            frozen_p = model.get_trainable_params()
            model.unfreeze_backbone()
            unfrozen_p = model.get_trainable_params()
            assert frozen_p["trainable"] < unfrozen_p["trainable"]
            print(f"[PASS] Freeze/unfreeze: {frozen_p['trainable']:,} -> {unfrozen_p['trainable']:,} trainable params")
            passed += 1
        except Exception as e:
            print(f"[FAIL] Freeze/unfreeze: {e}")
            failed += 1

    # ── Test 7: Inference API (demo mode) ───────────────────────────────
    try:
        from inference.predict import DRPredictor
        predictor = DRPredictor(cfg)
        result = predictor.predict(img)
        assert "grade" in result
        assert 0 <= result["grade"] <= 4
        assert 0 <= result["confidence"] <= 1
        assert len(result["probabilities"]) == 5
        assert abs(sum(result["probabilities"]) - 1.0) < 1e-3
        print(f"[PASS] Inference API: grade={result['grade']} ({result['severity']}), "
              f"conf={result['confidence']:.3f}, demo={result['demo_mode']}")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Inference API: {e}")
        import traceback; traceback.print_exc()
        failed += 1

    # ── Test 8: Grad-CAM placeholder ────────────────────────────────────
    try:
        from explainability.gradcam import GradCAMGenerator
        gcam_gen = GradCAMGenerator(model=None)
        overlay, is_real = gcam_gen.generate_heatmap(
            tensor_val.unsqueeze(0), img, predicted_grade=2
        )
        assert overlay is not None
        assert not is_real  # Should be placeholder since no model loaded
        print(f"[PASS] Grad-CAM placeholder: overlay size={overlay.size}, real={is_real}")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Grad-CAM: {e}")
        failed += 1

    # ── Test 9: Class weights ────────────────────────────────────────────
    try:
        from dataset.aptos_loader import APTOSLoader
        loader = APTOSLoader(cfg)
        fake_labels = np.array([0]*1000 + [1]*200 + [2]*500 + [3]*100 + [4]*150)
        weights = loader.get_class_weights(fake_labels)
        assert len(weights) == 5
        assert weights.min() > 0
        print(f"[PASS] Class weights (balanced): {[f'{w:.3f}' for w in weights]}")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Class weights: {e}")
        failed += 1

    # ── Summary ─────────────────────────────────────────────────────────
    print("=" * 65)
    print(f"  Results: {passed}/{passed+failed} passed  |  {failed} failed")
    if failed == 0:
        print("  ALL PIPELINE VERIFICATION TESTS PASSED [OK]")
    else:
        print(f"  {failed} test(s) failed — see above for details")
    print("=" * 65 + "\n")

    return failed == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
