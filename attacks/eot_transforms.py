import random
import torch


def apply_photometric_eot(
    images: torch.Tensor,
    color_jitter: float,
    noise_std: float,
) -> torch.Tensor:
    """
    Differentiable photometric augmentation for EOT.

    Applies random brightness, contrast, saturation scaling and additive
    Gaussian noise.  Sampled scalars are drawn fresh each call so every EOT
    sample sees a different augmentation.

    Args:
        images: [B, C, H, W] float tensor in [0, 1].
        color_jitter: half-range for brightness / contrast / saturation
                      multipliers, e.g. 0.2 → multipliers in [0.8, 1.2].
                      Pass 0 to disable.
        noise_std: std of additive Gaussian noise.  Pass 0 to disable.

    Returns:
        Augmented images clamped to [0, 1], same shape as input.
        Differentiable w.r.t. `images` (and therefore w.r.t. any patch
        pixels embedded in `images`).
    """
    if color_jitter > 0:
        brightness = 1.0 + random.uniform(-color_jitter, color_jitter)
        images = images * brightness

        contrast = 1.0 + random.uniform(-color_jitter, color_jitter)
        mean = images.mean(dim=(-2, -1), keepdim=True)
        images = (images - mean) * contrast + mean

        saturation = 1.0 + random.uniform(-color_jitter, color_jitter)
        gray = images.mean(dim=1, keepdim=True)
        images = gray + (images - gray) * saturation

    if noise_std > 0:
        images = images + torch.randn_like(images) * noise_std

    return torch.clamp(images, 0.0, 1.0)
