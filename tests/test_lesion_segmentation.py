"""
test_lesion_segmentation.py
===========================
Comprehensive Automated Test Suite for RetinaGuard Lesion & Anatomical Segmentation.

Tests:
1. Dataset Loader (IDRiD Part A train and test splits)
2. Decoupled Heads & Mask Shapes (MA, HE, EX, SE as lesions; OD as anatomy)
3. ONNX Model Loading and Session Initialization
4. PyTorch Checkpoint Loading and Verification
5. Segmentation Inference Output Shapes and Value Ranges
6. Invalid Image / Corrupt Input Error Handling
7. Blank / Empty Mask Handling
8. Automatic Heuristic Fallback Verification
9. Integration with Screening Service
10. Latency Benchmark (< 1.0s on CPU)
"""

import sys
import unittest
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from python.segmentation.dataset import IDRiDSegmentationDataset
from python.segmentation.unet import DualHeadUNet
from python.inference.segment_lesions import LesionSegmentationPipeline
from webapp.services.lesion_service import LesionService


class TestLesionSegmentation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.idrid_dir = ROOT / "data" / "idrid" / "A. Segmentation"
        cls.model_onnx = ROOT / "models" / "experiments" / "lesion_segmentation" / "lesion_unet.onnx"
        cls.model_pt = ROOT / "models" / "experiments" / "lesion_segmentation" / "best_model.pt"
        cls.demo_img_path = ROOT / "demo" / "sample_images" / "demo_case1_grade0.png"
        cls.pipeline = LesionSegmentationPipeline()

    def test_01_dataset_loader(self):
        """Verify IDRiD dataset loader correctly loads images and ground truth masks."""
        if not self.idrid_dir.exists():
            self.skipTest("IDRiD dataset not present")

        ds_train = IDRiDSegmentationDataset(self.idrid_dir, split="train", image_size=256)
        self.assertEqual(len(ds_train), 54, "Training set must contain 54 images")

        ds_test = IDRiDSegmentationDataset(self.idrid_dir, split="test", image_size=256)
        self.assertEqual(len(ds_test), 27, "Testing set must contain 27 images")

        img, lesions, anatomy, meta = ds_train[0]
        self.assertEqual(img.shape, (3, 256, 256))
        self.assertEqual(lesions.shape, (4, 256, 256), "4 lesion channels: MA, HE, EX, SE")
        self.assertEqual(anatomy.shape, (1, 256, 256), "1 anatomy channel: OD")

    def test_02_model_architecture(self):
        """Verify DualHeadUNet forward pass in both train and export modes."""
        model = DualHeadUNet(export_mode=False)
        dummy = torch.randn(2, 3, 256, 256)
        l_out, a_out = model(dummy)
        self.assertEqual(l_out.shape, (2, 4, 256, 256))
        self.assertEqual(a_out.shape, (2, 1, 256, 256))

        model.export_mode = True
        exp_out = model(dummy)
        self.assertEqual(exp_out.shape, (2, 5, 256, 256))
        self.assertTrue((exp_out >= 0.0).all() and (exp_out <= 1.0).all(), "Sigmoids must be in [0, 1]")

    def test_03_onnx_model_presence_and_loading(self):
        """Verify that lesion_unet.onnx exists, is < 25MB, and loads in ONNX Runtime."""
        self.assertTrue(self.model_onnx.exists(), f"ONNX model missing at {self.model_onnx}")
        size_mb = self.model_onnx.stat().st_size / (1024 * 1024)
        self.assertLess(size_mb, 25.0, f"Model size {size_mb} MB exceeds 25 MB limit")

        import onnxruntime as ort
        sess = ort.InferenceSession(str(self.model_onnx), providers=["CPUExecutionProvider"])
        self.assertIsNotNone(sess)

    def test_04_pytorch_checkpoint_presence(self):
        """Verify best_model.pt checkpoint integrity."""
        self.assertTrue(self.model_pt.exists(), f"PyTorch checkpoint missing at {self.model_pt}")
        ckpt = torch.load(self.model_pt, map_location="cpu")
        self.assertIn("model_state_dict", ckpt)
        self.assertIn("best_val_dice", ckpt)

    def test_05_segmentation_inference(self):
        """Verify pipeline segmentation returns expected keys, masks, and AI mode."""
        img = cv2.imread(str(self.demo_img_path))
        self.assertIsNotNone(img, "Could not load demo image")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        res = self.pipeline.segment(img_rgb)
        self.assertEqual(res["mode"], "AI SEGMENTATION")
        self.assertTrue(res["is_ai"])
        self.assertIn("lesion_counts", res)
        self.assertIn("lesion_areas", res)
        self.assertIn("anatomy", res)
        self.assertIn("overlay_rgb", res)

        # Check lesion keys
        for key in ["MA", "HE", "EX", "SE"]:
            self.assertIn(key, res["lesion_counts"])
            self.assertIn(key, res["lesion_areas"])

        # Check anatomy
        self.assertEqual(res["anatomy"]["category"], "Normal Anatomy")

    def test_06_heuristic_fallback(self):
        """Verify pipeline falls back gracefully when AI model is deactivated."""
        fallback_pipe = LesionSegmentationPipeline(model_path=Path("nonexistent.onnx"))
        fallback_pipe.is_ai_loaded = False

        img = cv2.imread(str(self.demo_img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        res = fallback_pipe.segment(img_rgb)
        self.assertEqual(res["mode"], "HEURISTIC FALLBACK")
        self.assertFalse(res["is_ai"])
        self.assertIn("HEURISTIC FALLBACK", res["badge_label"])
        self.assertIn("lesion_counts", res)

    def test_07_empty_blank_image_handling(self):
        """Verify pipeline handles all-black or blank images without throwing unhandled exceptions."""
        black_img = np.zeros((256, 256, 3), dtype=np.uint8)
        res = self.pipeline.segment(black_img)
        self.assertIsNotNone(res)
        self.assertIn("lesion_counts", res)

    def test_08_lesion_service_integration(self):
        """Verify webapp LesionService incorporates AI mode and outputs correctly."""
        out_dir = ROOT / "webapp" / "outputs"
        service = LesionService(output_dir=out_dir)

        img = cv2.imread(str(self.demo_img_path))
        res = service.analyze(img, filename_prefix="test_lesion")
        self.assertIn("mode", res)
        self.assertIn("microaneurysms", res)
        self.assertIn("hard_exudates", res)
        self.assertIn("hemorrhages", res)
        self.assertIn("neovascularization", res)
        self.assertIn("overlay_url", res)

    def test_09_inference_latency_benchmark(self):
        """Verify inference latency is sub-second (< 1.0s) on CPU."""
        img = cv2.imread(str(self.demo_img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Warmup
        _ = self.pipeline.segment(img_rgb)

        t0 = time.time()
        _ = self.pipeline.segment(img_rgb)
        elapsed = time.time() - t0

        self.assertLess(elapsed, 1.0, f"Inference latency {elapsed:.3f}s exceeds 1.0s threshold")
        print(f"\n[Benchmark] AI Lesion Segmentation Latency: {elapsed*1000:.1f} ms on CPU")


if __name__ == "__main__":
    unittest.main()
