"""
test_live_server.py
===================
Tests live running server on http://127.0.0.1:8000
"""

import urllib.request
import urllib.parse
import json

BASE_URL = "http://127.0.0.1:8000"

print("1. Health Endpoint:")
with urllib.request.urlopen(f"{BASE_URL}/health") as resp:
    data = json.loads(resp.read().decode("utf-8"))
    print("   Response:", data)
    assert data["status"] == "ok"
    assert data["model"] == "EXP-001"

print("\n2. Web UI Homepage:")
with urllib.request.urlopen(f"{BASE_URL}/") as resp:
    html = resp.read().decode("utf-8")
    print(f"   Status: {resp.status}, Bytes: {len(html)}")
    assert "RetinaGuard" in html
    assert "AI-Assisted Retinal Screening Workstation" in html

print("\n3. Demo Cases API:")
with urllib.request.urlopen(f"{BASE_URL}/api/demo-cases") as resp:
    demos = json.loads(resp.read().decode("utf-8"))
    print(f"   Count: {len(demos)}")
    for d in demos:
        print(f"   • {d['id']}: {d['label']} ({d['reference_name']})")
    assert len(demos) == 6

print("\n4. Full Screening on Demo Case 3 (Moderate Reference):")
req_data = urllib.parse.urlencode({"demo_id": "case_3"}).encode("utf-8")
req = urllib.request.Request(f"{BASE_URL}/api/analyze", data=req_data, method="POST")
with urllib.request.urlopen(req) as resp:
    res = json.loads(resp.read().decode("utf-8"))
    print(f"   Quality Score   : {res['quality']['score']}/100 ({res['quality']['status']})")
    print(f"   Safety Gate     : {res['safety_gate_triggered']} (Screening: {res['screening_status']})")
    print(f"   Landmarks       : Optic Disc={res['structures']['optic_disc']['centroid']}, Fovea={res['structures']['fovea']['coordinates']}, Vessel Density={res['structures']['vessels']['density_percent']}%")
    print(f"   Lesion Detection: MAs={res['lesions']['microaneurysms']['candidate_count']}, Exudates={res['lesions']['hard_exudates']['candidate_count']}, Hemorrhages={res['lesions']['hemorrhages']['candidate_count']}, NV Risk={res['lesions']['neovascularization']['risk_level']}")
    print(f"   EXP-001 Grading : Grade {res['grading']['grade']} ({res['grading']['severity_name']})")
    print(f"   Confidence      : {res['grading']['confidence_percent']}%")
    print(f"   Referable DR    : {res['grading']['referral']} ({res['grading']['referral_badge']})")
    print(f"   Grad-CAM Status : {res['grading'].get('gradcam', {})}")
    print(f"   Report URL      : {res['report']['relative_url']}")

print("\n4b. Dedicated Grad-CAM Request on Demo Case 3:")
req_gcam = urllib.parse.urlencode({"demo_id": "case_3", "target_grade": 2}).encode("utf-8")
req_g = urllib.request.Request(f"{BASE_URL}/api/gradcam", data=req_gcam, method="POST")
with urllib.request.urlopen(req_g) as resp_g:
    res_g = json.loads(resp_g.read().decode("utf-8"))
    print(f"   Grad-CAM Avail  : {res_g.get('gradcam_available')}")
    print(f"   Grad-CAM URL    : {res_g.get('gradcam_url')}")
    assert res_g.get("gradcam_available") is True

print("\n5. Testing Ungradable Demo Case 6 (Safety Gate):")
req_data6 = urllib.parse.urlencode({"demo_id": "case_6"}).encode("utf-8")
req6 = urllib.request.Request(f"{BASE_URL}/api/analyze", data=req_data6, method="POST")
with urllib.request.urlopen(req6) as resp:
    res6 = json.loads(resp.read().decode("utf-8"))
    print(f"   Quality Status  : {res6['quality']['status']} (Score: {res6['quality']['score']}/100)")
    print(f"   Safety Gate     : {res6['safety_gate_triggered']}")
    print(f"   Grading Output  : {res6['grading']} (DOWNSTREAM GRADING HALTED)")
    print(f"   Recapture Action: {res6['action_required']}")
    assert res6["safety_gate_triggered"] is True
    assert res6["grading"] is None

print("\nALL LIVE SERVER TESTS PASSED ON http://127.0.0.1:8000!")
