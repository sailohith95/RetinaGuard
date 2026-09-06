"""
RetinaGuard — Grad-CAM Explainability
=======================================
Generates gradient-weighted class activation maps for the trained model.

Uses PyTorch hooks to capture gradients and activations from the last
convolutional block of EfficientNet-B0. Produces a heatmap overlaid on
the original retinal image.

If the model is not available, returns a clearly labelled placeholder.

Usage:
    from explainability.gradcam import GradCAM
    gcam = GradCAM(model, target_layer_name="backbone.0.8")
    heatmap, overlay = gcam.generate(image_tensor, target_class=predicted_grade)
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

logger = logging.getLogger(__name__)


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for EfficientNet-B0.

    Registers forward/backward hooks on the target convolutional layer
    to capture feature maps and gradients without modifying the model.
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.device = next(model.parameters()).device

        # Auto-detect the last conv layer for EfficientNet-B0
        if target_layer is None:
            target_layer = self._find_last_conv(model)

        self._activations = None
        self._gradients = None

        # Register hooks
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, Optional[Image.Image]]:
        """
        Generate Grad-CAM heatmap.

        Args:
            input_tensor : (1, 3, H, W) preprocessed image tensor
            target_class : class index for gradient computation (None → argmax)

        Returns:
            heatmap_np    : (H, W) float32 heatmap in [0, 1]
            overlay_pil   : PIL image of heatmap blended over original
        """
        self.model.eval()
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad_()

        # Forward
        logits = self.model(input_tensor)
        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        # Backward w.r.t. target class
        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward()

        # Grad-CAM computation
        # gradients: (1, C, h, w) → mean over spatial
        gradients = self._gradients.detach().cpu()        # (1, C, h, w)
        activations = self._activations.detach().cpu()    # (1, C, h, w)

        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = torch.relu(cam)                                    # keep positives

        # Normalize to [0, 1]
        cam_np = cam.squeeze().numpy()
        if cam_np.max() > 0:
            cam_np = cam_np / cam_np.max()

        return cam_np, None  # overlay built by caller with original image

    def overlay_on_image(
        self,
        original_pil: Image.Image,
        heatmap_np: np.ndarray,
        alpha: float = 0.45,
        colormap: str = "jet",
    ) -> Image.Image:
        """
        Blend Grad-CAM heatmap over original retinal image.
        Uses fast, low-memory OpenCV blending (no float64 copies).

        Args:
            original_pil : original PIL image
            heatmap_np   : (h, w) float32 in [0, 1]
            alpha        : heatmap opacity
            colormap     : colormap name (defaults to jet)

        Returns:
            PIL Image with heatmap overlay
        """
        import cv2
        orig_w, orig_h = original_pil.size
        # Clamp overlay base to at most 1024 to prevent memory exhaustion
        max_dim = max(orig_w, orig_h)
        if max_dim > 1024:
            scale = 1024.0 / max_dim
            orig_to_blend = original_pil.resize((int(orig_w * scale), int(orig_h * scale)), Image.BILINEAR)
        else:
            orig_to_blend = original_pil

        w, h = orig_to_blend.size
        heatmap_uint8 = np.uint8(np.clip(heatmap_np * 255.0, 0, 255))
        heatmap_resized = cv2.resize(heatmap_uint8, (w, h), interpolation=cv2.INTER_LINEAR)
        heatmap_bgr = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        orig_np = np.array(orig_to_blend.convert("RGB"), dtype=np.uint8)
        blended = cv2.addWeighted(orig_np, 1.0 - alpha, heatmap_rgb, alpha, 0)
        return Image.fromarray(blended)

    def remove_hooks(self):
        """Call this when done to prevent memory leaks."""
        if hasattr(self, "_fwd_hook") and self._fwd_hook is not None:
            try:
                self._fwd_hook.remove()
            except Exception:
                pass
            self._fwd_hook = None
        if hasattr(self, "_bwd_hook") and self._bwd_hook is not None:
            try:
                self._bwd_hook.remove()
            except Exception:
                pass
            self._bwd_hook = None

    # ──────────────────────────────────────────────────────────────────────────
    # Hook callbacks
    # ──────────────────────────────────────────────────────────────────────────

    def _save_activation(self, module, input, output):
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    # ──────────────────────────────────────────────────────────────────────────
    # Auto-detect target layer
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _find_last_conv(model: nn.Module) -> nn.Module:
        """Find the last Conv2d layer before global average pooling."""
        last_conv = None
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        if last_conv is None:
            raise ValueError("No Conv2d layer found in model.")
        logger.debug("Grad-CAM target layer: %s", last_conv)
        return last_conv


class GradCAMGenerator:
    """
    High-level wrapper that manages the GradCAM lifecycle.
    Used by the inference pipeline and UI.
    """

    def __init__(self, model: Optional[nn.Module] = None, cfg: Optional[dict] = None):
        self.model = model
        self.cfg = cfg
        self._gcam: Optional[GradCAM] = None
        self.available = model is not None

    def generate_heatmap(
        self,
        image_tensor: torch.Tensor,
        original_pil: Image.Image,
        predicted_grade: int,
    ) -> Tuple[Optional[Image.Image], bool]:
        """
        Generate heatmap overlay.

        Returns:
            (overlay_image, is_real_gradcam)
            is_real_gradcam = False if this is a placeholder
        """
        if not self.available or self.model is None:
            return self._placeholder_heatmap(original_pil, predicted_grade), False

        try:
            if self._gcam is None:
                self._gcam = GradCAM(self.model)

            heatmap_np, _ = self._gcam.generate(image_tensor, target_class=predicted_grade)
            overlay = self._gcam.overlay_on_image(original_pil, heatmap_np, alpha=0.45)
            return overlay, True

        except Exception as e:
            logger.warning("Grad-CAM failed (%s); returning placeholder.", e)
            return self._placeholder_heatmap(original_pil, predicted_grade), False

    def cleanup(self):
        if self._gcam is not None:
            self._gcam.remove_hooks()
            self._gcam = None

    @staticmethod
    def _placeholder_heatmap(original_pil: Image.Image, grade: int) -> Image.Image:
        """
        DEMO VISUALIZATION — Gaussian blob placeholder.
        Clearly NOT a real Grad-CAM. Shown only when model is unavailable.
        """
        import matplotlib.cm as cm
        from PIL import Image as PILImage

        w, h = original_pil.size
        Y, X = np.mgrid[0:h, 0:w]
        # Grade-dependent position heuristic (purely illustrative)
        cx = w * (0.35 + 0.05 * grade)
        cy = h * (0.45 + 0.05 * grade)
        sigma = min(w, h) * 0.28
        gaussian = np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))
        gaussian = gaussian / gaussian.max()

        cmap = cm.get_cmap("jet")
        heat_rgb = np.uint8(cmap(gaussian)[:, :, :3] * 255)
        orig_arr = np.array(original_pil.convert("RGB")).astype(float)
        blended = 0.55 * orig_arr + 0.45 * heat_rgb.astype(float)
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        result = PILImage.fromarray(blended)
        return result
