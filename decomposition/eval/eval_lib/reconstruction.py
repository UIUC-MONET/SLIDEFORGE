"""M1: Render-and-compare reconstruction fidelity.

Per metrics_survey.md: composite mask-cut crops onto a blank canvas at their bboxes,
then compare with the original slide. Lower is better for MSE / LPIPS; higher is
better for SSIM / PSNR / CLIP-cosine.

LPIPS and CLIP are *optional*. If their libraries aren't importable in this env,
the metric is reported as `null` with a `_skipped` reason — MSE / SSIM / PSNR
always run.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from PIL import Image

from .common import SlideEvalSample, build_reconstruction, load_original_rgb


# ---------- always-available metrics ---------------------------------------------------

def _mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = _mse(a, b)
    if mse <= 1e-9:
        return 99.0
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    from skimage.metrics import structural_similarity as sk_ssim
    return float(sk_ssim(a, b, channel_axis=2, data_range=255))


# ---------- optional metrics -----------------------------------------------------------

import threading
_init_lock = threading.Lock()

_LPIPS_MODEL = None
_LPIPS_SKIP_REASON: str | None = None
_DEVICE = "cpu"  # will be set to cuda if available


def _get_device() -> str:
    """Detect GPU availability. Prefer cuda:1 to avoid contention with GPU 0."""
    global _DEVICE
    try:
        import torch
        if torch.cuda.is_available():
            _DEVICE = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
        else:
            _DEVICE = "cpu"
    except Exception:
        _DEVICE = "cpu"
    return _DEVICE


def pre_init_models() -> None:
    """Warm up the optional LPIPS and CLIP models sequentially and thread-safely."""
    global _LPIPS_MODEL, _LPIPS_SKIP_REASON, _CLIP_MODEL, _CLIP_SKIP_REASON, _DEVICE
    _get_device()
    print(f"[reconstruction] Using device: {_DEVICE}", flush=True)
    with _init_lock:
        if not _LPIPS_SKIP_REASON and _LPIPS_MODEL is None:
            try:
                import torch
                import lpips
                _LPIPS_MODEL = lpips.LPIPS(net="alex", verbose=False).eval().to(_DEVICE)
            except Exception as e:
                _LPIPS_SKIP_REASON = f"lpips unavailable: {type(e).__name__}: {e}"

        if not _CLIP_SKIP_REASON and _CLIP_MODEL is None:
            try:
                import open_clip
                import torch
                model, _, preprocess = open_clip.create_model_and_transforms(
                    "ViT-B-32", pretrained="laion2b_s34b_b79k"
                )
                model.eval().to(_DEVICE)
                _CLIP_MODEL = (model, preprocess)
            except Exception as e:
                _CLIP_SKIP_REASON = f"open_clip unavailable: {type(e).__name__}: {e}"


def _try_lpips(a: np.ndarray, b: np.ndarray) -> tuple[float | None, str | None]:
    global _LPIPS_MODEL, _LPIPS_SKIP_REASON
    if _LPIPS_SKIP_REASON:
        return None, _LPIPS_SKIP_REASON
    with _init_lock:
        if _LPIPS_MODEL is None:
            try:
                import torch
                import lpips  # noqa
                _LPIPS_MODEL = lpips.LPIPS(net="alex", verbose=False).eval().to(_DEVICE)
            except Exception as e:
                _LPIPS_SKIP_REASON = f"lpips unavailable: {type(e).__name__}: {e}"
                return None, _LPIPS_SKIP_REASON
    import torch
    def _to_t(x):
        t = torch.from_numpy(x).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
        return t.to(_DEVICE)
    with torch.no_grad():
        v = _LPIPS_MODEL(_to_t(a), _to_t(b)).item()
    return float(v), None


_CLIP_MODEL: Any = None
_CLIP_SKIP_REASON: str | None = None


def _try_clip(a: np.ndarray, b: np.ndarray) -> tuple[float | None, str | None]:
    """Cosine similarity of CLIP image embeddings. Returns (score, skip_reason)."""
    global _CLIP_MODEL, _CLIP_SKIP_REASON
    if _CLIP_SKIP_REASON:
        return None, _CLIP_SKIP_REASON
    with _init_lock:
        if _CLIP_MODEL is None:
            try:
                import open_clip  # noqa
                import torch
                model, _, preprocess = open_clip.create_model_and_transforms(
                    "ViT-B-32", pretrained="laion2b_s34b_b79k"
                )
                model.eval().to(_DEVICE)
                _CLIP_MODEL = (model, preprocess)
            except Exception as e:
                _CLIP_SKIP_REASON = f"open_clip unavailable: {type(e).__name__}: {e}"
                return None, _CLIP_SKIP_REASON
    import torch
    model, preprocess = _CLIP_MODEL
    with torch.no_grad():
        ea = model.encode_image(preprocess(Image.fromarray(a)).unsqueeze(0).to(_DEVICE))
        eb = model.encode_image(preprocess(Image.fromarray(b)).unsqueeze(0).to(_DEVICE))
        ea = ea / ea.norm(dim=-1, keepdim=True)
        eb = eb / eb.norm(dim=-1, keepdim=True)
        return float((ea @ eb.T).item()), None


# ---------- public API -----------------------------------------------------------------

def evaluate(
    sample: SlideEvalSample,
    save_reconstruction_to: str | None = None,
) -> dict[str, Any]:
    original = load_original_rgb(sample)
    reconstruction = build_reconstruction(sample)
    if save_reconstruction_to:
        os.makedirs(os.path.dirname(save_reconstruction_to), exist_ok=True)
        Image.fromarray(reconstruction).save(save_reconstruction_to)

    result: dict[str, Any] = {
        "mse": _mse(original, reconstruction),
        "psnr": _psnr(original, reconstruction),
        "ssim": _ssim(original, reconstruction),
        "num_final_components": len(sample.final_components),
    }
    lpips_score, lpips_skip = _try_lpips(original, reconstruction)
    result["lpips"] = lpips_score
    if lpips_skip:
        result["lpips_skipped"] = lpips_skip

    clip_score, clip_skip = _try_clip(original, reconstruction)
    result["clip_cosine"] = clip_score
    if clip_skip:
        result["clip_skipped"] = clip_skip

    if save_reconstruction_to:
        result["reconstruction_path"] = save_reconstruction_to
    return result


def aggregate(slide_results: list[dict[str, Any]]) -> dict[str, Any]:
    if not slide_results:
        return {}
    keys = ["mse", "psnr", "ssim", "lpips", "clip_cosine"]
    agg: dict[str, Any] = {}
    for k in keys:
        vals = [r[k] for r in slide_results if r.get(k) is not None]
        if vals:
            agg[f"mean_{k}"] = float(np.mean(vals))
            agg[f"std_{k}"] = float(np.std(vals))
            agg[f"n_{k}"] = len(vals)
    return agg
