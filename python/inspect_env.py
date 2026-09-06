import os
import json
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

files = [
    "launch.m",
    "runAllTests.m",
    "app/RetinaGuardApp.m",
    "core/screening/runScreeningPipeline.m",
    "core/quality/assessImageQuality.m",
    "core/enhancement/enhanceRetinalImage.m",
    "core/structures/detectStructures.m",
    "core/lesions/detectLesions.m",
    "core/grading/gradeDR.m",
    "core/grading/runONNXModel.m",
    "core/grading/runPythonInference.m",
    "models/aptos_efficientnet/best_model.onnx",
    "models/aptos_efficientnet/best_model.pt",
    "models/experiments/EXP-001/best_model.pt",
    "python/explainability/gradcam.py",
    "core/reporting/generateReport.m",
    "demo/getDemoCases.m",
    "python/test_end_to_end_sih.py"
]

print("=== 1. FILE EXISTENCE CHECK ===")
missing = 0
for f in files:
    full_p = ROOT / f
    status = "EXISTS" if full_p.exists() else "MISSING"
    if not full_p.exists():
        missing += 1
    print(f"{f:<45}: {status}")

print(f"\nTotal verified files: {len(files)}, Missing: {missing}")

print("\n=== 2. PRODUCTION MODEL INTEGRITY CHECK ===")
def get_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        while chunk := fp.read(65536):
            h.update(chunk)
    return h.hexdigest()

pt_prod = ROOT / "models/aptos_efficientnet/best_model.pt"
pt_exp1 = ROOT / "models/experiments/EXP-001/best_model.pt"
onnx_prod = ROOT / "models/aptos_efficientnet/best_model.onnx"

hash_prod = get_hash(pt_prod)
hash_exp1 = get_hash(pt_exp1)

print(f"Production model path: {pt_prod}")
print(f"Production SHA-256:   {hash_prod}")
print(f"EXP-001 SHA-256:      {hash_exp1}")
print(f"Hashes identical:     {hash_prod == hash_exp1}")
print(f"ONNX model exists:    {onnx_prod.exists()} (size: {onnx_prod.stat().st_size:,} bytes)")

meta_file = ROOT / "models/aptos_efficientnet/metadata.json"
metrics_file = ROOT / "models/aptos_efficientnet/metrics.json"

if meta_file.exists():
    with open(meta_file) as fp:
        print("\nMetadata snapshot:")
        print(json.dumps(json.load(fp), indent=2))

if metrics_file.exists():
    with open(metrics_file) as fp:
        print("\nMetrics snapshot:")
        print(json.dumps(json.load(fp), indent=2))
