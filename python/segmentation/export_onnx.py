"""
export_onnx.py
==============
Exports DualHeadUNet best checkpoint to ONNX format.
"""

import sys
import os
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from python.segmentation.unet import DualHeadUNet

EXP_DIR = ROOT / "models" / "experiments" / "lesion_segmentation"
CKPT_PATH = EXP_DIR / "best_model.pt"
ONNX_PATH = EXP_DIR / "lesion_unet.onnx"

def export():
    print(f"[*] Loading checkpoint: {CKPT_PATH}")
    model = DualHeadUNet(
        in_channels=3,
        num_lesion_classes=4,
        num_anatomy_classes=1,
        base_features=32,
        export_mode=True
    )

    checkpoint = torch.load(CKPT_PATH, map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.eval()

    dummy_input = torch.randn(1, 3, 256, 256)
    
    print(f"[*] Exporting to: {ONNX_PATH}...")
    torch.onnx.export(
        model,
        dummy_input,
        str(ONNX_PATH),
        input_names=["input_image"],
        output_names=["sigmoid_masks"],
        dynamic_axes={
            "input_image": {0: "batch_size"},
            "sigmoid_masks": {0: "batch_size"}
        },
        opset_version=14,
        dynamo=False
    )

    size_mb = ONNX_PATH.stat().st_size / (1024 * 1024)
    print(f"[OK] ONNX model successfully exported: {ONNX_PATH} ({size_mb:.2f} MB)")

    # Verify ONNX model with onnxruntime
    import onnxruntime as ort
    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    res = sess.run(None, {"input_image": dummy_input.numpy()})
    print(f"[OK] ONNX Runtime verification pass! Output shape: {res[0].shape}")


if __name__ == "__main__":
    export()
