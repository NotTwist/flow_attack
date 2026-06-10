#!/usr/bin/env python3
"""
Run depth estimation (Depth-Anything-V2) and semantic segmentation (PSPNet-Cityscapes)
on a single image and save a side-by-side figure.

Usage:
  python scripts/infer_single_image.py --image /path/to/image.jpg
  python scripts/infer_single_image.py --image /path/to/image.jpg --out results/my_image
"""

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'flow_library'))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

import ptlflow
import ptlflow.utils.io_adapter
import torchvision.transforms.functional as tvf

from models.model_utils import load_mde_model, load_seg_model
from attacks.adversarial_manhole.adv_manhole.texture_mapping.depth_utils import disp_to_depth


# Cityscapes 19-class palette + names
_CS_PALETTE = np.array([
    [128,  64, 128], [244,  35, 232], [ 70,  70,  70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170,  30], [220, 220,   0],
    [107, 142,  35], [152, 251, 152], [ 70, 130, 180], [220,  20,  60],
    [255,   0,   0], [  0,   0, 142], [  0,   0,  70], [  0,  60, 100],
    [  0,  80, 100], [  0,   0, 230], [119,  11,  32],
], dtype=np.uint8)

_CS_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign", "vegetation",
    "terrain", "sky", "person", "rider", "car", "truck",
    "bus", "train", "motorcycle", "bicycle",
]


def make_legend(palette, classes, height):
    """Vertical color legend, resized to `height` pixels."""
    import cv2
    n = len(classes)
    swatch_h = max(20, height // n)
    legend = np.zeros((n * swatch_h, 160, 3), dtype=np.uint8)
    for i, (color, name) in enumerate(zip(palette, classes)):
        y0, y1 = i * swatch_h, (i + 1) * swatch_h
        legend[y0:y1] = color
        cv2.putText(legend, name, (6, y0 + swatch_h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255) if color.mean() < 128 else (0, 0, 0), 1)
    return cv2.resize(legend, (160, height))


def prepare_input(image_path, device):
    """Load image → ptlflow-style inputs dict with a dummy second frame."""
    img = Image.open(image_path).convert('RGB')
    t   = tvf.to_tensor(img).unsqueeze(0).to(device)        # [1,3,H,W]
    # ptlflow expects a 5-D tensor: [B, 2, C, H, W]
    images = torch.stack([t, t], dim=1)                     # [1,2,3,H,W]
    B, _, C, H, W = images.shape
    dummy_flow  = torch.zeros(B, 1, 2, H, W, device=device)
    dummy_valid = torch.ones(B, 1, 1, H, W,  device=device)
    # minimal io_adapter — we only need the images dict key
    inputs = {'images': images, 'flows': dummy_flow, 'valids': dummy_valid}
    return inputs, t.squeeze(0)   # inputs dict, raw [3,H,W] tensor


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--image', required=True, help='Input image path (jpg/png)')
    ap.add_argument('--mde_model', default='depth-anything-v2')
    ap.add_argument('--ss_model',  default='pspnet_cityscapes')
    ap.add_argument('--out', default='',
                    help='Output prefix (default: same dir as input, same stem)')
    args = ap.parse_args()

    if not os.path.isfile(args.image):
        sys.exit(f"File not found: {args.image}")

    out_prefix = args.out or os.path.splitext(args.image)[0]
    os.makedirs(os.path.dirname(out_prefix) or '.', exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("Loading models…")
    mde_model = load_mde_model(model_name=args.mde_model, device=device)
    ss_model  = load_seg_model(model_name=args.ss_model,  device=device)
    for p in ss_model.parameters():
        p.requires_grad_(False)

    print(f"Processing {args.image}…")
    inputs, img_t = prepare_input(args.image, device)

    with torch.no_grad():
        raw_depth = mde_model(inputs)
        ss_logits = ss_model(inputs, return_logits=True)

    # ── depth ─────────────────────────────────────────────────────────────────
    d = raw_depth
    if d.dim() == 2:   d = d.unsqueeze(0).unsqueeze(0)
    elif d.dim() == 3: d = d.unsqueeze(1)
    # metric depth via disparity conversion
    metric_d = disp_to_depth(d.float())
    d_np = metric_d.squeeze().cpu().numpy()
    d_np = np.nan_to_num(d_np, nan=0.0, posinf=0.0, neginf=0.0)

    # ── segmentation ──────────────────────────────────────────────────────────
    seg_cls = ss_logits.argmax(dim=1).squeeze().cpu().numpy().astype(np.int32)
    seg_rgb = _CS_PALETTE[np.clip(seg_cls, 0, len(_CS_PALETTE) - 1)]

    # ── original image ────────────────────────────────────────────────────────
    img_np = img_t.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
    H, W   = img_np.shape[:2]

    # ── figure ────────────────────────────────────────────────────────────────
    import cv2
    legend = make_legend(_CS_PALETTE, _CS_CLASSES, H)
    seg_with_legend = np.concatenate([seg_rgb, legend], axis=1)

    depth_norm = ((d_np - d_np.min()) / (d_np.max() - d_np.min() + 1e-8)).clip(0, 1)
    depth_rgb  = (plt.get_cmap('inferno')(depth_norm)[:, :, :3] * 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(24, 5))
    axes[0].imshow(img_np);       axes[0].set_title('Input image',     fontsize=13, fontweight='bold')
    axes[1].imshow(depth_rgb);    axes[1].set_title('Depth (Depth-Anything-V2)', fontsize=13, fontweight='bold')
    axes[2].imshow(seg_with_legend); axes[2].set_title('Segmentation (PSPNet Cityscapes)', fontsize=13, fontweight='bold')
    for ax in axes:
        ax.axis('off')
    plt.tight_layout(pad=0.5)

    out_fig = out_prefix + '_depth_seg.png'
    fig.savefig(out_fig, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved → {out_fig}")

    # also save individual maps
    out_depth = out_prefix + '_depth.png'
    out_seg   = out_prefix + '_seg.png'
    Image.fromarray(depth_rgb).save(out_depth)
    cv2.imwrite(out_seg, cv2.cvtColor(seg_with_legend, cv2.COLOR_RGB2BGR))
    print(f"Depth  → {out_depth}")
    print(f"Seg    → {out_seg}")


if __name__ == '__main__':
    main()
