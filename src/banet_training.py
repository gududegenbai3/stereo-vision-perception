"""BANet supervised loss, training mode and fixed-domain evaluation interfaces."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .banet import padding_for_shape
from .stereo_training_data import StereoTrainingSample

if TYPE_CHECKING:
    import torch


def supervised_loss(predictions: list[torch.Tensor], target: torch.Tensor, valid: torch.Tensor,
                    max_disparity: int, auxiliary_weight: float) -> torch.Tensor:
    """Masked SmoothL1 final+auxiliary loss, float32 (B,1,H,W) px tensors.

    Respect dataset validity in addition to disparity range. Unlike the upstream
    training script, do not ignore its supplied validity argument for sparse GT.
    Invalid targets (including inf) are excluded before arithmetic. Empty labels
    or incompatible tensor shapes raise ValueError instead of yielding NaN loss.
    """
    import torch
    import torch.nn.functional as functional

    if len(predictions) != 2 or valid.shape != target.shape or any(p.shape != target.shape for p in predictions):
        raise ValueError("Expected two predictions and GT/mask of shape (B,1,H,W)")
    if target.ndim != 4 or target.shape[1] != 1 or valid.dtype != torch.bool:
        raise ValueError("Expected single-channel disparities and bool validity")
    mask: torch.Tensor = valid & torch.isfinite(target) & (target > 0) & (target < max_disparity)
    if not mask.any():
        raise ValueError("No valid supervised pixels")
    return (functional.smooth_l1_loss(predictions[0][mask], target[mask])
            + auxiliary_weight * functional.smooth_l1_loss(predictions[1][mask], target[mask]))


def set_training_mode(model: torch.nn.Module, freeze_batch_norm_statistics: bool) -> None:
    """Train all unfrozen parameters; optionally retain pretrained BN running stats."""
    import torch

    model.train()
    if freeze_batch_norm_statistics:
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()


def predict_raw(model: torch.nn.Module, sample: StereoTrainingSample, max_disparity: int) -> np.ndarray:
    """Return raw float32 (H,W) px predictions, no GT-dependent/output filtering."""
    import torch
    import torch.nn.functional as functional

    height, width = sample.left_rgb.shape[:2]
    padding: tuple[int, int, int, int] = padding_for_shape(height, width)
    left_pad, _, top_pad, _ = padding
    tensors: list[torch.Tensor] = []
    for image in (sample.left_rgb, sample.right_rgb):
        value: torch.Tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).unsqueeze(0).float().to("cuda")
        tensors.append(functional.pad(value, padding, mode="replicate"))
    model.eval()
    with torch.inference_mode():
        output: torch.Tensor = model(tensors[0], tensors[1], max_disparity)
        prediction: np.ndarray = output[0, 0, top_pad:top_pad + height, left_pad:left_pad + width].float().cpu().numpy().copy()
    if prediction.shape != (height, width) or not np.isfinite(prediction).all():
        raise ValueError(f"Invalid dense network output: {sample.name}")
    return prediction


def fixed_domain_metrics(prediction: np.ndarray, sample: StereoTrainingSample, max_disparity: int) -> dict:
    """Fixed non-occluded GT domain; report Bad1/Bad2 and pixel-pooled sufficient sums."""
    reference: np.ndarray = sample.non_occluded_mask & (sample.disparity_px > 0) & (sample.disparity_px < max_disparity)
    if prediction.shape != reference.shape or not np.isfinite(prediction[reference]).all() or not reference.any():
        raise ValueError("Invalid prediction or empty evaluation domain")
    error: np.ndarray = np.abs(prediction[reference].astype(np.float64) - sample.disparity_px[reference].astype(np.float64))
    return {"sample": sample.name, "group": sample.group, "pixels": int(error.size),
        "absolute_error_sum_px": float(error.sum()), "bad1_pixels": int((error > 1).sum()), "bad2_pixels": int((error > 2).sum()),
        "mae_px": float(error.mean()), "bad1_fraction": float((error > 1).mean()), "bad2_fraction": float((error > 2).mean()),
        "excluded_non_occluded_pixels": int(sample.non_occluded_mask.sum() - error.size)}


def pool_training_metrics(rows: list[dict]) -> dict:
    """Pool by fixed GT pixel counts rather than averaging unequal image sizes."""
    count: int = sum(row["pixels"] for row in rows)
    if count == 0:
        raise ValueError("Empty pooled evaluation")
    return {"pixels": count, "mae_px": sum(row["absolute_error_sum_px"] for row in rows) / count,
        "bad1_fraction": sum(row["bad1_pixels"] for row in rows) / count,
        "bad2_fraction": sum(row["bad2_pixels"] for row in rows) / count}
