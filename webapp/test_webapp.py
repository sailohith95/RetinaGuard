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
        cls.grade0_path = SAMPLE_DIR / "demo_grade0.png"
        cls.grade1_path = SAMPLE_DIR / "demo_grade1.png"
        cls.grade2_path = SAMPLE_DIR / "demo_grade2.png"
        cls.grade3_path = SAMPLE_DIR / "demo_grade3.png"
        cls.grade4_path = SAMPLE_DIR / "demo_grade4.png"
        for p in [cls.grade0_path, cls.grade1_path, cls.grade2_path, cls.grade3_path, cls.grade4_path]:
            assert p.exists(), f"Missing {p}"

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
        with open(self.grade0_path, "rb") as fp:
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
        resp = client.post("/api/analyze", data={"demo_id": "grade_0"})
        self.assertEqual(resp.status_code, 200)
        q = resp.json()["quality"]
        self.assertIn("score", q)
        self.assertIn("status", q)
        self.assertIn("focus", q)
        self.assertIn("illumination", q)
        self.assertIn("contrast", q)
        self.assertGreater(q["score"], 50.0)
        self.assertEqual(q["status"], "GOOD")

    # 6. Ungradable safety gate (tested on uploaded degraded image)
    def test_06_ungradable_safety_gate(self):
        bad_img = np.zeros((224, 224, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".png", bad_img)
        files = {"file": ("dark_ungradable.png", io.BytesIO(buf.tobytes()), "image/png")}
        resp = client.post("/api/analyze", files=files)
        self.assertEqual(resp.status_code, 200)
        res = resp.json()
        self.assertTrue(res["safety_gate_triggered"])
        self.assertEqual(res["screening_status"], "UNGRADABLE")
        self.assertIsNone(res["grading"], "Grading MUST be halted for ungradable images")
        self.assertIn("Recapture", res["quality"]["recommendation"])

    # 7. Enhancement
    def test_07_enhancement(self):
        resp = client.post("/api/analyze", data={"demo_id": "grade_2"})
        self.assertEqual(resp.status_code, 200)
        enh = resp.json()["enhancement"]
        self.assertIsNotNone(enh)
        self.assertIn("enhanced_url", enh)
        self.assertTrue(enh["enhanced_url"].startswith("/outputs/"))
        self.assertGreater(len(enh["steps_applied"]), 0)

    # 8. Structure analysis
    def test_08_structure_analysis(self):
        resp = client.post("/api/analyze", data={"demo_id": "grade_0"})
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
        resp = client.post("/api/analyze", data={"demo_id": "grade_2"})
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
        resp = client.post("/api/analyze", data={"demo_id": "grade_0"})
        self.assertEqual(resp.status_code, 200)
        g = resp.json()["grading"]
        self.assertIn(g["grade"], [0, 1, 2, 3, 4])
        self.assertEqual(len(g["probabilities"]), 5)
        prob_sum = sum(p["probability"] for p in g["probabilities"])
        self.assertAlmostEqual(prob_sum, 1.0, delta=0.01)
        self.assertEqual(g["model_version"], "EXP-001")

    # 11. Grad-CAM
    def test_11_gradcam(self):
        resp = client.post("/api/analyze", data={"demo_id": "grade_2"})
        self.assertEqual(resp.status_code, 200)
        g = resp.json()["grading"]
        self.assertIsNotNone(g["gradcam_url"])
        self.assertTrue(g["gradcam_url"].startswith("/outputs/"))

    # 12. Recommendation
    def test_12_recommendation(self):
        # Grade 2 -> Referable
        resp = client.post("/api/analyze", data={"demo_id": "grade_2"})
        self.assertEqual(resp.status_code, 200)
        g = resp.json()["grading"]
        if g["grade"] >= 2:
            self.assertTrue(g["referral"])
            self.assertEqual(g["referral_badge"], "REFERABLE DR")
        else:
            self.assertFalse(g["referral"])

    # 13. Report generation
    def test_13_report_generation(self):
        resp = client.post("/api/analyze", data={"demo_id": "grade_0"})
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
        self.assertEqual(len(cases), 5, "Curated demo section must contain exactly 5 demo options")
        case_ids = [c["id"] for c in cases]
        self.assertEqual(case_ids, ["grade_0", "grade_1", "grade_2", "grade_3", "grade_4"])
        self.assertNotIn("case_6", case_ids)

    # 15. Full end-to-end API flow
    def test_15_full_end_to_end_flow(self):
        # Step 1: Health
        h = client.get("/health")
        self.assertEqual(h.status_code, 200)
        
        # Step 2: Load demo cases
        c = client.get("/api/demo-cases")
        self.assertEqual(c.status_code, 200)
        self.assertEqual(len(c.json()), 5)
        
        # Step 3: Analyze Grade 1 (Mild NPDR)
        a1 = client.post("/api/analyze", data={"demo_id": "grade_1"})
        self.assertEqual(a1.status_code, 200)
        self.assertEqual(a1.json()["screening_status"], "GRADABLE")
        self.assertEqual(a1.json()["grading"]["grade"], 1)
        
        # Step 4: Analyze ungradable uploaded image (Safety gate)
        bad_img = np.zeros((224, 224, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".png", bad_img)
        files = {"file": ("dark.png", io.BytesIO(buf.tobytes()), "image/png")}
        a_bad = client.post("/api/analyze", files=files)
        self.assertEqual(a_bad.status_code, 200)
        self.assertEqual(a_bad.json()["screening_status"], "UNGRADABLE")
        self.assertTrue(a_bad.json()["safety_gate_triggered"])

    # 16. Demo case sequential switching: Grade 0 -> 1 -> 2 -> 3 -> 4 -> 0
    def test_16_demo_switching_sequence(self):
        # Grade 0
        r0 = client.post("/api/analyze", data={"demo_id": "grade_0"}).json()
        self.assertEqual(r0["screening_status"], "GRADABLE")
        self.assertEqual(r0["demo_reference"]["reference_grade"], 0)
        self.assertEqual(r0["grading"]["grade"], 0)

        # Grade 1
        r1 = client.post("/api/analyze", data={"demo_id": "grade_1"}).json()
        self.assertEqual(r1["screening_status"], "GRADABLE")
        self.assertEqual(r1["demo_reference"]["reference_grade"], 1)
        self.assertEqual(r1["grading"]["grade"], 1)

        # Grade 2
        r2 = client.post("/api/analyze", data={"demo_id": "grade_2"}).json()
        self.assertEqual(r2["screening_status"], "GRADABLE")
        self.assertEqual(r2["demo_reference"]["reference_grade"], 2)
        self.assertEqual(r2["grading"]["grade"], 2)

        # Grade 3
        r3 = client.post("/api/analyze", data={"demo_id": "grade_3"}).json()
        self.assertEqual(r3["screening_status"], "GRADABLE")
        self.assertEqual(r3["demo_reference"]["reference_grade"], 3)
        self.assertEqual(r3["grading"]["grade"], 3)

        # Grade 4
        r4 = client.post("/api/analyze", data={"demo_id": "grade_4"}).json()
        self.assertEqual(r4["screening_status"], "GRADABLE")
        self.assertEqual(r4["demo_reference"]["reference_grade"], 4)
        self.assertEqual(r4["grading"]["grade"], 4)

        # Grade 0 again (Confirm no state bleed/residue from previous Grade 4)
        r0_second = client.post("/api/analyze", data={"demo_id": "grade_0"}).json()
        self.assertEqual(r0_second["screening_status"], "GRADABLE")
        self.assertFalse(r0_second["safety_gate_triggered"])
        self.assertEqual(r0_second["demo_reference"]["reference_grade"], 0)
        self.assertEqual(r0_second["demo_reference"]["id"], "grade_0")
        self.assertEqual(r0_second["grading"]["grade"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
