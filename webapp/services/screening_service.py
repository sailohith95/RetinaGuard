"""
screening_service.py
====================
Master Orchestrator for RetinaGuard Screening Pipeline.
Coordinates Quality Gate, Enhancement, Landmarks, Lesions, EXP-001 Inference,
True Grad-CAM, and Clinical Reporting.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import cv2
from PIL import Image

from webapp.services.quality_service import QualityService
from webapp.services.enhancement_service import EnhancementService
from webapp.services.structure_service import StructureService
from webapp.services.lesion_service import LesionService
from webapp.services.grading_service import GradingService
from webapp.services.report_service import ReportService

ROOT = Path(__file__).resolve().parent.parent.parent


class ScreeningService:
    def __init__(self, uploads_dir: Path, outputs_dir: Path):
        self.uploads_dir = uploads_dir
        self.outputs_dir = outputs_dir
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

        # Initialize sub-services
        self.quality_service = QualityService()
        self.enhancement_service = EnhancementService(outputs_dir)
        self.structure_service = StructureService(outputs_dir)
        self.lesion_service = LesionService(outputs_dir)
        self.grading_service = GradingService(outputs_dir)
        self.report_service = ReportService(outputs_dir)

        # Demo cases directory
        self.demo_dir = ROOT / "demo" / "sample_images"

    def get_demo_cases(self) -> list:
        """Returns the 6 curated authentic demo screening cases."""
        all_cases = [
            {
                "id": "case_1",
                "label": "Case 1 — Normal Retina (No DR)",
                "reference_grade": 0,
                "reference_name": "No DR",
                "filename": "demo_case1_grade0.png",
                "image_url": "/demo/sample_images/demo_case1_grade0.png",
                "description": "Healthy retinal fundus with clear optic disc and macula. Expected negative screening."
            },
            {
                "id": "case_2",
                "label": "Case 2 — Mild NPDR Reference",
                "reference_grade": 1,
                "reference_name": "Mild NPDR",
                "filename": "demo_case2_grade1.png",
                "image_url": "/demo/sample_images/demo_case2_grade1.png",
                "description": "Early microvascular alterations with isolated microaneurysms. Non-referable clinical triage."
            },
            {
                "id": "case_3",
                "label": "Case 3 — Moderate NPDR Reference",
                "reference_grade": 2,
                "reference_name": "Moderate NPDR",
                "filename": "demo_case3_grade2.png",
                "image_url": "/demo/sample_images/demo_case3_grade2.png",
                "description": "Hard exudates and microaneurysms present. Referable DR threshold triggered (3–6 month referral)."
            },
            {
                "id": "case_4",
                "label": "Case 4 — Severe NPDR Reference",
                "reference_grade": 3,
                "reference_name": "Severe NPDR",
                "filename": "demo_case4_grade3.png",
                "image_url": "/demo/sample_images/demo_case4_grade3.png",
                "description": "Multiple intra-retinal hemorrhages and microvascular abnormalities. Prompt referral required (1 month)."
            },
            {
                "id": "case_5",
                "label": "Case 5 — Proliferative DR Reference",
                "reference_grade": 4,
                "reference_name": "Proliferative DR",
                "filename": "demo_case5_grade4.png",
                "image_url": "/demo/sample_images/demo_case5_grade4.png",
                "description": "High lesion burden with neovascularization risk. Referable to retina specialist."
            },
            {
                "id": "case_6",
                "label": "Case 6 — Poor Quality / Ungradable",
                "reference_grade": -1,
                "reference_name": "Ungradable",
                "filename": "demo_case6_ungradable.png",
                "image_url": "/demo/sample_images/demo_case6_ungradable.png",
                "description": "Severely blurred and underexposed fundus scan. Demonstrates patient safety quality gate."
            }
        ]
        return [c for c in all_cases if (self.demo_dir / c["filename"]).is_file()]

    def run_screening(
        self,
        image_path: Path,
        original_url: str,
        patient_id: str = "PT-82910",
        exam_id: str = "EX-00412",
        eye: str = "Right (OD)",
        demo_reference: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Runs the full 7-stage automated screening pipeline.
        """
        # Load image via OpenCV and PIL
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise ValueError(f"Unable to read image at {image_path}. File may be corrupted.")

        pil_img = Image.open(image_path).convert("RGB")
        h, w = img_bgr.shape[:2]

        # ---------------------------------------------------------------------
        # STAGE 1: IMAGE QUALITY ASSESSMENT
        # ---------------------------------------------------------------------
        quality = self.quality_service.assess(img_bgr)

        # ---------------------------------------------------------------------
        # CRITICAL SAFETY GATE: UNGRADABLE IMAGE REJECTION
        # ---------------------------------------------------------------------
        if not quality["gradable"]:
            result = {
                "patient_id": patient_id,
                "exam_id": exam_id,
                "eye": eye,
                "image_info": {
                    "width": w,
                    "height": h,
                    "original_url": original_url
                },
                "original_url": original_url,
                "demo_reference": demo_reference,
                "quality": quality,
                "safety_gate_triggered": True,
                "screening_status": "UNGRADABLE",
                "message": "DOWNSTREAM GRADING HALTED: Image quality is insufficient for diagnostic screening.",
                "action_required": quality["recommendation"],
                "grading": None,
                "enhancement": None,
                "structures": None,
                "lesions": None,
                "report": None
            }
            # Generate ungradable report
            report_res = self.report_service.generate(result, patient_id, exam_id, eye)
            result["report"] = report_res
            return result

        # ---------------------------------------------------------------------
        # STAGE 2: CLINICAL IMAGE ENHANCEMENT
        # ---------------------------------------------------------------------
        enhancement = self.enhancement_service.enhance(img_bgr)
        enhanced_bgr = enhancement["enhanced_bgr"]

        # ---------------------------------------------------------------------
        # STAGE 3: RETINAL ANATOMICAL LANDMARKS
        # ---------------------------------------------------------------------
        structures = self.structure_service.detect(enhanced_bgr)
        od_mask = structures["optic_disc"]["mask_np"]
        vessel_density = structures["vessels"]["density_percent"]

        # Remove numpy arrays from JSON output
        clean_structures = {
            "optic_disc": {
                "detected": structures["optic_disc"]["detected"],
                "centroid": structures["optic_disc"]["centroid"],
                "radius_px": structures["optic_disc"]["radius_px"],
                "methodology": structures["optic_disc"]["methodology"]
            },
            "fovea": structures["fovea"],
            "vessels": {
                "detected": structures["vessels"]["detected"],
                "density_percent": structures["vessels"]["density_percent"],
                "methodology": structures["vessels"]["methodology"]
            },
            "vessel_overlay_url": structures["vessel_overlay_url"],
            "structures_overlay_url": structures["structures_overlay_url"]
        }

        # ---------------------------------------------------------------------
        # STAGE 4: COMPUTER VISION LESION CANDIDATE ANALYSIS
        # ---------------------------------------------------------------------
        lesions = self.lesion_service.analyze(
            enhanced_bgr,
            od_mask=od_mask,
            vessel_density=vessel_density
        )

        # ---------------------------------------------------------------------
        # STAGE 5: DEEP LEARNING DR GRADING (EXP-001) & TRUE GRAD-CAM
        # ---------------------------------------------------------------------
        grading = self.grading_service.grade(pil_img)

        # ---------------------------------------------------------------------
        # STAGE 6: COMPILE CLINICAL RESULT OBJECT
        # ---------------------------------------------------------------------
        result = {
            "patient_id": patient_id,
            "exam_id": exam_id,
            "eye": eye,
            "image_info": {
                "width": w,
                "height": h,
                "original_url": original_url
            },
            "original_url": original_url,
            "demo_reference": demo_reference,
            "quality": quality,
            "safety_gate_triggered": False,
            "screening_status": "GRADABLE",
            "enhancement": {
                "enhanced_url": enhancement["relative_url"],
                "steps_applied": enhancement["steps_applied"]
            },
            "structures": clean_structures,
            "lesions": lesions,
            "grading": grading,
            "overlays": {
                "original": original_url,
                "enhanced": enhancement["relative_url"],
                "vessels": clean_structures["vessel_overlay_url"],
                "lesions": lesions["overlay_url"],
                "heatmap": grading["gradcam_url"]
            },
            "report": None
        }

        # ---------------------------------------------------------------------
        # STAGE 7: CLINICAL REPORT GENERATION
        # ---------------------------------------------------------------------
        report_res = self.report_service.generate(result, patient_id, exam_id, eye)
        result["report"] = report_res

        return result
