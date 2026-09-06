"""
test_webapp.py
==============
Automated Test Suite for RetinaGuard Standalone Localhost Web Application.

Tests at minimum:
1.  Health endpoint
2.  Homepage loads
3.  Valid image upload
4.  Invalid image rejection
5.  Quality assessment
6.  Ungradable safety gate
7.  Enhancement
8.  Structure analysis
9.  Lesion analysis
10. EXP-001 inference
11. Grad-CAM
12. Recommendation
13. Report generation
14. Demo case loading
15. Full end-to-end API flow
"""

import os
import sys
import unittest
from pathlib import Path
import io
import cv2
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from webapp.main import app

client = TestClient(app)
SAMPLE_DIR = ROOT / "demo" / "sample_images"


class TestRetinaGuardWebApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case1_path = SAMPLE_DIR / "demo_case1_grade0.png"
        cls.case3_path = SAMPLE_DIR / "demo_case3_grade2.png"
        cls.case6_path = SAMPLE_DIR / "demo_case6_ungradable.png"
        assert cls.case1_path.exists(), f"Missing {cls.case1_path}"
        assert cls.case3_path.exists(), f"Missing {cls.case3_path}"
        assert cls.case6_path.exists(), f"Missing {cls.case6_path}"

    # 1. Health endpoint
    def test_01_health_endpoint(self):
        resp = client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["model"], "EXP-001")
        self.assertTrue(data["offline"])
        self.assertAlmostEqual(data["qwk"], 0.7342, places=4)

    # 2. Homepage loads
    def test_02_homepage_loads(self):
        resp = client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("RetinaGuard", resp.text)
        self.assertIn("AI-Assisted Retinal Screening", resp.text)
        self.assertIn("EXP-001", resp.text)

    # 3. Valid image upload
    def test_03_valid_image_upload(self):
        with open(self.case1_path, "rb") as fp:
            files = {"file": ("test_fundus.png", fp, "image/png")}
            data = {"patient_id": "PT-TEST", "exam_id": "EX-TEST", "eye": "Right (OD)"}
            resp = client.post("/api/analyze", files=files, data=data)
            
        self.assertEqual(resp.status_code, 200)
        res = resp.json()
        self.assertEqual(res["patient_id"], "PT-TEST")
        self.assertIn("quality", res)
        self.assertIn("grading", res)

    # 4. Invalid image rejection
    def test_04_invalid_image_rejection(self):
        # A: Non-image file (.txt)
        txt_data = io.BytesIO(b"Hello world not a fundus image")
        files = {"file": ("document.txt", txt_data, "text/plain")}
        resp = client.post("/api/analyze", files=files)
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Invalid file format", resp.json()["detail"])

        # B: Empty file
        empty_data = io.BytesIO(b"")
        files = {"file": ("empty.png", empty_data, "image/png")}
        resp = client.post("/api/analyze", files=files)
        self.assertEqual(resp.status_code, 400)

    # 5. Quality assessment
    def test_05_quality_assessment(self):
        resp = client.post("/api/analyze", data={"demo_id": "case_1"})
        self.assertEqual(resp.status_code, 200)
        q = resp.json()["quality"]
        self.assertIn("score", q)
        self.assertIn("status", q)
        self.assertIn("focus", q)
        self.assertIn("illumination", q)
        self.assertIn("contrast", q)
        self.assertGreater(q["score"], 50.0)
        self.assertEqual(q["status"], "GOOD")

    # 6. Ungradable safety gate
    def test_06_ungradable_safety_gate(self):
        # Case 6 is ungradable
        resp = client.post("/api/analyze", data={"demo_id": "case_6"})
        self.assertEqual(resp.status_code, 200)
        res = resp.json()
        self.assertTrue(res["safety_gate_triggered"])
        self.assertEqual(res["screening_status"], "UNGRADABLE")
        self.assertIsNone(res["grading"], "Grading MUST be halted for ungradable images")
        self.assertIn("Recapture", res["quality"]["recommendation"])

    # 7. Enhancement
    def test_07_enhancement(self):
        resp = client.post("/api/analyze", data={"demo_id": "case_3"})
        self.assertEqual(resp.status_code, 200)
        enh = resp.json()["enhancement"]
        self.assertIsNotNone(enh)
        self.assertIn("enhanced_url", enh)
        self.assertTrue(enh["enhanced_url"].startswith("/outputs/"))
        self.assertGreater(len(enh["steps_applied"]), 0)

    # 8. Structure analysis
    def test_08_structure_analysis(self):
        resp = client.post("/api/analyze", data={"demo_id": "case_1"})
        self.assertEqual(resp.status_code, 200)
        st = resp.json()["structures"]
        self.assertTrue(st["optic_disc"]["detected"])
        self.assertEqual(len(st["optic_disc"]["centroid"]), 2)
        self.assertGreater(st["optic_disc"]["radius_px"], 0)
        self.assertEqual(len(st["fovea"]["coordinates"]), 2)
        self.assertGreater(st["vessels"]["density_percent"], 1.0)
        self.assertTrue(st["vessel_overlay_url"].startswith("/outputs/"))

    # 9. Lesion analysis
    def test_09_lesion_analysis(self):
        resp = client.post("/api/analyze", data={"demo_id": "case_3"})
        self.assertEqual(resp.status_code, 200)
        les = resp.json()["lesions"]
        self.assertIn("microaneurysms", les)
        self.assertIn("hard_exudates", les)
        self.assertIn("hemorrhages", les)
        self.assertIn("neovascularization", les)
        self.assertIn(les["neovascularization"]["risk_level"], ["Low", "Moderate", "High"])
        self.assertTrue(les["overlay_url"].startswith("/outputs/"))

    # 10. EXP-001 inference
    def test_10_exp001_inference(self):
        resp = client.post("/api/analyze", data={"demo_id": "case_1"})
        self.assertEqual(resp.status_code, 200)
        g = resp.json()["grading"]
        self.assertIn(g["grade"], [0, 1, 2, 3, 4])
        self.assertEqual(len(g["probabilities"]), 5)
        prob_sum = sum(p["probability"] for p in g["probabilities"])
        self.assertAlmostEqual(prob_sum, 1.0, delta=0.01)
        self.assertEqual(g["model_version"], "EXP-001")

    # 11. Decoupled Grad-CAM & Dedicated Endpoint
    def test_11_gradcam_decoupled_and_dedicated_endpoint(self):
        # A: Core screening returns immediately with Grad-CAM status ready
        resp = client.post("/api/analyze", data={"demo_id": "case_3"})
        self.assertEqual(resp.status_code, 200)
        res = resp.json()
        self.assertIn("grading", res)
        g = res["grading"]
        self.assertIn("gradcam", g)
        self.assertEqual(g["gradcam"]["endpoint"], "/api/gradcam")
        self.assertIsNone(res["overlays"]["heatmap"])

        # B: Dedicated /api/gradcam endpoint generates real PyTorch heatmap
        gcam_resp = client.post("/api/gradcam", data={"demo_id": "case_3", "target_grade": 2})
        self.assertEqual(gcam_resp.status_code, 200)
        gcam_data = gcam_resp.json()
        self.assertTrue(gcam_data["gradcam_available"])
        self.assertIsNotNone(gcam_data["gradcam_url"])
        self.assertTrue(gcam_data["gradcam_url"].startswith("/outputs/"))

    # 12. Recommendation
    def test_12_recommendation(self):
        # Case 3 -> Grade >= 2 (Referable)
        resp = client.post("/api/analyze", data={"demo_id": "case_3"})
        self.assertEqual(resp.status_code, 200)
        g = resp.json()["grading"]
        if g["grade"] >= 2:
            self.assertTrue(g["referral"])
            self.assertEqual(g["referral_badge"], "REFERABLE DR")
        else:
            self.assertFalse(g["referral"])

    # 13. Report generation
    def test_13_report_generation(self):
        resp = client.post("/api/analyze", data={"demo_id": "case_1"})
        self.assertEqual(resp.status_code, 200)
        rep = resp.json()["report"]
        self.assertIsNotNone(rep)
        self.assertTrue(rep["filename"].endswith(".html"))
        
        # Test download endpoint
        dl_resp = client.get(f"/api/download-report/{rep['filename']}")
        self.assertEqual(dl_resp.status_code, 200)
        self.assertIn("RetinaGuard", dl_resp.text)
        self.assertIn("IMPORTANT CLINICAL DISCLAIMER", dl_resp.text)

    # 14. Demo case loading
    def test_14_demo_case_loading(self):
        resp = client.get("/api/demo-cases")
        self.assertEqual(resp.status_code, 200)
        cases = resp.json()
        self.assertEqual(len(cases), 6)
        case_ids = [c["id"] for c in cases]
        self.assertIn("case_1", case_ids)
        self.assertIn("case_6", case_ids)

    # 15. Full end-to-end API flow
    def test_15_full_end_to_end_flow(self):
        # Step 1: Health
        h = client.get("/health")
        self.assertEqual(h.status_code, 200)
        
        # Step 2: Load demo cases
        c = client.get("/api/demo-cases")
        self.assertEqual(c.status_code, 200)
        
        # Step 3: Analyze Case 2 (Mild reference)
        a2 = client.post("/api/analyze", data={"demo_id": "case_2"})
        self.assertEqual(a2.status_code, 200)
        self.assertEqual(a2.json()["screening_status"], "GRADABLE")
        
        # Step 4: Analyze Case 6 (Ungradable safety gate)
        a6 = client.post("/api/analyze", data={"demo_id": "case_6"})
        self.assertEqual(a6.status_code, 200)
        self.assertEqual(a6.json()["screening_status"], "UNGRADABLE")
        self.assertTrue(a6.json()["safety_gate_triggered"])

    # 16. Regression: Screening API returns a response promptly without hanging
    def test_16_regression_screening_api_returns_response(self):
        import time
        t0 = time.time()
        resp = client.post("/api/analyze", data={"demo_id": "case_1"})
        elapsed = time.time() - t0
        self.assertEqual(resp.status_code, 200)
        self.assertLess(elapsed, 5.0, "Decoupled screening API must complete well under 5s")
        data = resp.json()
        self.assertIn("grading", data)
        self.assertIn("grade", data["grading"])

    # 17. Regression: Backend exceptions return structured error and do not hang
    def test_17_regression_backend_exceptions_no_hang(self):
        # Unknown demo case ID -> must return 400
        resp1 = client.post("/api/analyze", data={"demo_id": "nonexistent_demo_xyz"})
        self.assertEqual(resp1.status_code, 400)
        self.assertIn("Unknown demo case ID", resp1.json()["detail"])

        # Empty POST body -> must return 400
        resp2 = client.post("/api/analyze", data={})
        self.assertEqual(resp2.status_code, 400)
        self.assertIn("No fundus image provided", resp2.json()["detail"])

    # 18. Regression: Grad-CAM failure does not return 500 error
    def test_18_regression_gradcam_failure_fallback(self):
        from unittest.mock import patch
        from explainability.gradcam import GradCAM

        # Mock GradCAM.generate to throw a simulated RuntimeError on /api/gradcam
        with patch.object(GradCAM, "generate", side_effect=RuntimeError("Simulated GradCAM hook failure")):
            resp = client.post("/api/gradcam", data={"demo_id": "case_1", "target_grade": 0})
            self.assertEqual(resp.status_code, 200, "Endpoint must return 200 with structured failure")
            data = resp.json()
            self.assertFalse(data["gradcam_available"])
            self.assertIsNone(data["gradcam_url"])
            self.assertIn("error", data)

    # 19. Regression: High resolution image clamping and memory safety
    def test_19_regression_large_image_clamping(self):
        img = cv2.imread(str(self.case1_path))
        # Create 1568x1568 high-res image by tiling to preserve high-frequency features
        large_img = np.tile(img, (7, 7, 1))
        _, enc = cv2.imencode(".png", large_img)
        files = {"file": ("large_fundus.png", io.BytesIO(enc.tobytes()), "image/png")}
        resp = client.post("/api/analyze", files=files)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["image_info"]["width"], 1568)
        self.assertEqual(data["image_info"]["height"], 1568)
        self.assertEqual(data["screening_status"], "GRADABLE")
        self.assertIsNotNone(data["grading"])
        self.assertIn("grade", data["grading"])

    # 20. Regression: Frontend code contains AbortController timeout & error clearing
    def test_20_regression_frontend_timeout_and_error_handling(self):
        js_path = ROOT / "webapp" / "static" / "js" / "app.js"
        self.assertTrue(js_path.exists())
        js_content = js_path.read_text(encoding="utf-8")
        self.assertIn("AbortController", js_content, "Frontend must use AbortController")
        self.assertIn("45000", js_content, "Frontend must configure screening timeout")
        self.assertIn("processingOverlay.style.display = \"none\"", js_content, "Overlay must be cleared in finally")

    # 21. Regression: Grad-CAM timeout guard catches slow execution gracefully
    def test_21_gradcam_timeout_fallback(self):
        from unittest.mock import patch
        import time

        def slow_generate(*args, **kwargs):
            time.sleep(0.15)
            return None

        with patch("explainability.gradcam.GradCAM.generate", side_effect=slow_generate):
            from webapp.main import screening_service, DEMO_DIR
            sample_path = DEMO_DIR / "demo_case1_grade0.png"
            res = screening_service.run_gradcam(sample_path, target_grade=0, timeout_sec=0.02)
            self.assertFalse(res["gradcam_available"])
            self.assertIn("budget", res["error"])

    # 22. Safety Gate: Ungradable scans completely bypass Grad-CAM
    def test_22_ungradable_bypasses_gradcam(self):
        resp = client.post("/api/analyze", data={"demo_id": "case_6"})
        self.assertEqual(resp.status_code, 200)
        res = resp.json()
        self.assertTrue(res["safety_gate_triggered"])
        self.assertIsNone(res["grading"], "Grading must be null for ungradable images")
        self.assertIsNone(res.get("overlays"), "Overlays must be absent or null for ungradable images")


if __name__ == "__main__":
    unittest.main(verbosity=2)
