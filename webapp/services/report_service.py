"""
report_service.py
=================
Generates professional, printable, self-contained HTML clinical screening reports.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Any


class ReportService:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, result: Dict[str, Any], patient_id: str = "PT-82910", exam_id: str = "EX-00412", eye: str = "Right (OD)") -> Dict[str, str]:
        date_str = datetime.now().strftime("%d-%b-%Y %H:%M")
        timestamp_slug = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        q = result.get("quality") or {}
        enh = result.get("enhancement") or {}
        st = result.get("structures") or {}
        l = result.get("lesions") or {}
        g = result.get("grading") or {}
        is_ungradable = not q.get("gradable", True)

        # Grade color mapping
        grade = g.get("grade", -1)
        colors = {
            0: "#1a7f37",
            1: "#b08800",
            2: "#b08800",
            3: "#cf222e",
            4: "#82071e",
            -1: "#6e7781"
        }
        grade_color = colors.get(grade, "#6e7781")

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>RetinaGuard Screening Report — {exam_id}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #f6f8fa; color: #1f2328; font-size: 13px; line-height: 1.5; }}
  .sheet {{ max-width: 860px; margin: 24px auto; background: #ffffff; border: 1px solid #d0d7de; border-radius: 6px; overflow: hidden; box-shadow: 0 4px 12px rgba(0,0,0,0.06); }}
  .header {{ background: #0f2137; color: #ffffff; padding: 20px 32px; display: flex; justify-content: space-between; align-items: center; border-bottom: 3px solid #1f6feb; }}
  .header h1 {{ font-size: 22px; font-weight: 700; letter-spacing: 0.5px; }}
  .header .sub {{ font-size: 11px; color: #8cb4f5; text-transform: uppercase; letter-spacing: 1px; margin-top: 2px; }}
  .header .meta {{ text-align: right; font-size: 11px; color: #d0d7de; }}
  .section {{ padding: 18px 32px; border-bottom: 1px solid #e1e4e8; }}
  .sec-title {{ font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 1.2px; color: #57606a; margin-bottom: 12px; }}
  .grid-3 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  .kv-label {{ font-size: 10px; text-transform: uppercase; color: #656d76; font-weight: 600; }}
  .kv-val {{ font-size: 13px; font-weight: 600; color: #1f2328; margin-top: 2px; }}
  .badge-good {{ background: #dafbe1; color: #1a7f37; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 700; }}
  .badge-ungradable {{ background: #ffebe9; color: #cf222e; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 700; }}
  .badge-borderline {{ background: #fff8c5; color: #9a6700; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 700; }}
  .grade-card {{ background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 6px; padding: 18px; text-align: center; margin-bottom: 14px; }}
  .grade-big {{ font-size: 32px; font-weight: 800; color: {grade_color}; }}
  .grade-name {{ font-size: 15px; font-weight: 700; margin-top: 4px; }}
  .conf-bar {{ display: flex; align-items: center; justify-content: center; gap: 8px; font-size: 12px; color: #57606a; margin-top: 6px; }}
  .referral-box {{ background: #ddf4ff; border-left: 4px solid #0969da; padding: 12px 16px; border-radius: 0 4px 4px 0; margin-top: 12px; }}
  .referral-title {{ font-size: 11px; font-weight: 700; text-transform: uppercase; color: #0969da; margin-bottom: 4px; }}
  .table {{ width: 100%; border-collapse: collapse; margin-top: 6px; font-size: 12px; }}
  .table th {{ text-align: left; padding: 6px 10px; background: #f6f8fa; border-bottom: 1px solid #d0d7de; font-size: 11px; color: #57606a; }}
  .table td {{ padding: 7px 10px; border-bottom: 1px solid #eaeef2; }}
  .img-grid {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin-top: 10px; }}
  .img-card {{ border: 1px solid #d0d7de; border-radius: 4px; overflow: hidden; background: #000; text-align: center; }}
  .img-card img {{ width: 100%; height: 180px; object-fit: contain; display: block; }}
  .img-card .caption {{ background: #f6f8fa; padding: 5px; font-size: 10px; font-weight: 600; color: #57606a; border-top: 1px solid #d0d7de; }}
  .disclaimer {{ background: #fff8c5; border-left: 4px solid #bf8700; padding: 12px 16px; font-size: 11px; color: #656d76; margin: 18px 32px; border-radius: 0 4px 4px 0; }}
  .footer {{ background: #f6f8fa; padding: 12px 32px; text-align: center; font-size: 11px; color: #656d76; border-top: 1px solid #d0d7de; }}
  @media print {{ body {{ background: #fff; }} .sheet {{ border: none; box-shadow: none; margin: 0; width: 100%; }} }}
</style>
</head>
<body>

<div class="sheet">
  <div class="header">
    <div>
      <h1>RetinaGuard</h1>
      <div class="sub">AI-Assisted Retinal Screening Report &bull; Decision Support</div>
    </div>
    <div class="meta">
      <div>Report Date: {date_str}</div>
      <div>Production Model: EXP-001 (EfficientNet-B0)</div>
    </div>
  </div>

  <div class="section">
    <div class="sec-title">Examination & Patient Information</div>
    <div class="grid-3">
      <div><div class="kv-label">Patient Identifier</div><div class="kv-val">{patient_id}</div></div>
      <div><div class="kv-label">Examination ID</div><div class="kv-val">{exam_id}</div></div>
      <div><div class="kv-label">Eye Orientation</div><div class="kv-val">{eye}</div></div>
    </div>
  </div>

  <div class="section">
    <div class="sec-title">Image Quality Assessment</div>
    <div class="grid-2">
      <div>
        <div class="kv-label">Quality Status</div>
        <div class="kv-val">
          <span class="badge-{q.get('status', 'ungradable').lower()}">{q.get('status', 'UNGRADABLE')}</span>
          &nbsp;&bull;&nbsp; Quality Score: <strong>{q.get('score', 0)} / 100</strong>
        </div>
        <div style="font-size: 11px; color: #57606a; margin-top: 6px;">
          Reason: {q.get('reason', 'N/A')}
        </div>
      </div>
      <div>
        <table class="table" style="margin: 0;">
          <tr><td>Focus / Sharpness</td><td><strong>{q.get('focus', 'N/A')}</strong> (Laplacian: {q.get('laplacian_var', 'N/A')})</td></tr>
          <tr><td>Illumination</td><td><strong>{q.get('illumination', 'N/A')}</strong> (Mean: {q.get('mean_lum', 'N/A')})</td></tr>
          <tr><td>Contrast & FOV</td><td><strong>{q.get('contrast', 'N/A')}</strong> &bull; FOV: {q.get('fov_percent', 'N/A')}%</td></tr>
        </table>
      </div>
    </div>
  </div>
"""

        if is_ungradable:
            html += f"""
  <div class="section" style="background: #fff5f5;">
    <div class="sec-title" style="color: #cf222e;">Safety Gate Triggered &mdash; Screening Halted</div>
    <div style="padding: 16px; border: 1px solid #ffdcd7; border-radius: 6px; background: #ffebe9; color: #82071e;">
      <strong>DOWNSTREAM GRADING HALTED:</strong> Image quality is insufficient for reliable deep learning evaluation.
      To prevent medical misdiagnosis, automated DR grading is disabled.
      <div style="margin-top: 8px; font-weight: 600;">Clinical Action: {q.get('recommendation', 'Recapture image.')}</div>
    </div>
  </div>
"""
        else:
            html += f"""
  <div class="section">
    <div class="sec-title">Diabetic Retinopathy Screening Result</div>
    <div class="grade-card">
      <div style="font-size: 11px; text-transform: uppercase; color: #57606a; font-weight: 700;">AI Screening Severity</div>
      <div class="grade-big">Level {g.get('grade', 'N/A')}</div>
      <div class="grade-name">{g.get('severity_name', 'N/A')}</div>
      <div class="conf-bar">
        Model Confidence: <strong>{g.get('confidence_percent', 0)}%</strong> &bull;
        Status: <strong>{g.get('referral_badge', 'N/A')}</strong>
      </div>
    </div>

    <div class="referral-box">
      <div class="referral-title">Clinical Screening Recommendation</div>
      <div style="font-weight: 600; color: #0969da;">{g.get('recommendation', 'N/A')}</div>
    </div>
  </div>

  <div class="section">
    <div class="sec-title">Quantitative Computer-Vision Findings</div>
    <div class="grid-2">
      <div>
        <div style="font-weight: 700; font-size: 11px; margin-bottom: 4px; color: #57606a;">Retinal Landmarks</div>
        <table class="table">
          <tr><td>Optic Disc</td><td>Centroid: {st.get('optic_disc', {}).get('centroid', 'N/A')}, Radius: {st.get('optic_disc', {}).get('radius_px', 'N/A')} px</td></tr>
          <tr><td>Fovea Centralis</td><td>Coordinates: {st.get('fovea', {}).get('coordinates', 'N/A')}</td></tr>
          <tr><td>Retinal Vasculature</td><td>Vascular Density: <strong>{st.get('vessels', {}).get('density_percent', 'N/A')}%</strong></td></tr>
        </table>
      </div>
      <div>
        <div style="font-weight: 700; font-size: 11px; margin-bottom: 4px; color: #57606a;">Lesion Candidate Analysis</div>
        <table class="table">
          <tr><td>Microaneurysms</td><td><strong>{l.get('microaneurysms', {}).get('candidate_count', 0)}</strong> candidates</td></tr>
          <tr><td>Hard Exudates</td><td><strong>{l.get('hard_exudates', {}).get('candidate_count', 0)}</strong> candidates ({l.get('hard_exudates', {}).get('total_area_px', 0)} px, Disc-masked)</td></tr>
          <tr><td>Hemorrhages</td><td><strong>{l.get('hemorrhages', {}).get('candidate_count', 0)}</strong> candidates ({l.get('hemorrhages', {}).get('total_area_px', 0)} px)</td></tr>
          <tr><td>Neovascularization Risk</td><td><strong>{l.get('neovascularization', {}).get('risk_level', 'Low')}</strong> ({l.get('neovascularization', {}).get('description', '')})</td></tr>
        </table>
      </div>
    </div>
  </div>

  <div class="section">
    <div class="sec-title">Multi-Layer Visual Evidence</div>
    <div class="img-grid">
      <div class="img-card">
        <img src="{result.get('original_url', '')}" alt="Original Fundus">
        <div class="caption">Original Fundus Scan</div>
      </div>
      <div class="img-card">
        <img src="{l.get('overlay_url', '')}" alt="Lesion Candidate Overlay">
        <div class="caption">Lesion Candidates Overlay</div>
      </div>
      <div class="img-card">
        <img src="{g.get('gradcam_url', '')}" alt="Grad-CAM Saliency Map">
        <div class="caption">Grad-CAM (Model Attention)</div>
      </div>
    </div>
  </div>
"""

        html += f"""
  <div class="disclaimer">
    <strong>IMPORTANT CLINICAL DISCLAIMER:</strong> This report is generated by RetinaGuard (Version 1.0.0), an AI-assisted screening decision-support prototype. It is NOT an autonomous medical diagnosis. All findings, candidate lesion counts, and recommendations must be reviewed and confirmed by a licensed ophthalmologist or retina specialist before clinical management or treatment decisions are undertaken.
  </div>

  <div class="footer">
    RetinaGuard &mdash; AI-Assisted Diabetic Retinopathy Screening Workstation &bull; SIH 2026
  </div>
</div>

</body>
</html>"""

        filename = f"report_{exam_id}_{timestamp_slug}.html"
        out_path = self.output_dir / filename
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)

        return {
            "filename": filename,
            "file_path": str(out_path),
            "relative_url": f"/outputs/{filename}"
        }
