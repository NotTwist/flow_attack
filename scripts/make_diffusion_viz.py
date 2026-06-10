#!/usr/bin/env python3
"""
Produce the cherry-pick 4×2 visualization using a pre-trained diffusion patch
(no model download required — uses the patch already saved in the project).

Applies the patch to a KITTI-15 image pair, runs RAFT / Depth-Anything-V2 /
PSPNet, and saves a publication-ready figure.

Usage:
  python scripts/make_diffusion_viz.py --sample_idx 12 --patch_projection
  python scripts/make_diffusion_viz.py --sweep 0 5 10 15 20
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
from torch.utils.data import DataLoader, Subset

import ptlflow
import ptlflow.utils.io_adapter

from datasets_utils.datasets import KITTI
from datasets_utils.dataset_utils import load_dataset_args
from models.model_utils import load_mde_model, load_seg_model
from utils.process_images import _to_numpy_image, _to_numpy_flow, replace_images_dic
from utils.seed import set_seed
from attacks.DetectionDefenses.helper_functions.patch_adversary import PatchAdversary
from attacks.patch_projection import (
    project_patch_on_scene, fit_plane_from_depth, keep_largest_component,
)
from attacks.adversarial_manhole.adv_manhole.texture_mapping.depth_utils import disp_to_depth
from flow_library.flow_plot import colorplot_light


# Default diffusion patch — epoch 3 from the existing project run
_DEFAULT_PATCH = os.path.join(
    _ROOT, 'experiments', 'quick_physical_3task_diffusion',
    'patch_checkpoints', 'patch_epoch_003.png',
)

# Cityscapes 19-class palette (RGB)
_CS_PALETTE = np.array([
    [128,  64, 128], [244,  35, 232], [ 70,  70,  70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170,  30], [220, 220,   0],
    [107, 142,  35], [152, 251, 152], [ 70, 130, 180], [220,  20,  60],
    [255,   0,   0], [  0,   0, 142], [  0,   0,  70], [  0,  60, 100],
    [  0,  80, 100], [  0,   0, 230], [119,  11,  32],
], dtype=np.uint8)


# ── rendering ────────────────────────────────────────────────────────────────

def flow_to_rgb(flow_tensor):
    """Per-panel auto-scale so ego-motion fills the color range."""
    if torch.is_tensor(flow_tensor):
        flow = flow_tensor.detach().cpu().float().numpy()
    else:
        flow = np.asarray(flow_tensor, dtype=np.float32)
    if flow.ndim == 4:
        flow = flow[0]
    hw2 = np.nan_to_num(np.transpose(flow, (1, 2, 0)), nan=0.0)
    return colorplot_light(hw2, auto_scale=True, return_max=False).astype(np.uint8)


def depth_to_rgb(d_np, vmin, vmax):
    d = np.nan_to_num(d_np, nan=0.0, posinf=0.0, neginf=0.0)
    d_norm = ((d - vmin) / (vmax - vmin + 1e-8)).clip(0.0, 1.0)
    return (plt.get_cmap('inferno')(d_norm)[:, :, :3] * 255).astype(np.uint8)


def seg_to_rgb(seg_tensor):
    if torch.is_tensor(seg_tensor):
        seg = seg_tensor.detach().cpu().numpy()
    else:
        seg = np.asarray(seg_tensor)
    if seg.ndim == 3:
        seg = seg[0]
    return _CS_PALETTE[np.clip(seg.astype(np.int32), 0, len(_CS_PALETTE) - 1)]


def img_to_uint8(t):
    return (_to_numpy_image(t) * 255).clip(0, 255).astype(np.uint8)


def np_depth(t):
    d = t.detach().cpu().float().numpy() if torch.is_tensor(t) else np.asarray(t, np.float32)
    while d.ndim > 2:
        d = d[0]
    return np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)


# ── core ─────────────────────────────────────────────────────────────────────

def load_models(model_name, mde_name, ss_name, device):
    of_model = ptlflow.get_model(model_name, 'chairs').to(device)
    of_model.eval()
    for p in of_model.parameters():
        p.requires_grad_(False)
    mde_model = load_mde_model(model_name=mde_name, device=device)
    ss_model  = load_seg_model(model_name=ss_name,  device=device)
    for p in ss_model.parameters():
        p.requires_grad_(False)
    return of_model, mde_model, ss_model


def run_one(idx, full_ds, of_model, mde_model, ss_model, patch_path,
            patch_size, patch_projection, device):
    """Apply the patch to sample `idx` and return rendered panels."""
    loader = DataLoader(Subset(full_ds, [idx]), batch_size=1, shuffle=False)
    img_size = (full_ds.image_x_dim, full_ds.image_y_dim)

    A = PatchAdversary(
        patch_path,
        size=patch_size,
        angle=0, scale=1,
        change_of_variable=False,
        random_location=False,
        image_size=img_size,
        ellipse_scale_y=3.0,
    ).to(device)

    for images, flow_gt, valid, meta, K in loader:
        break

    io_adapter = ptlflow.utils.io_adapter.IOAdapter(
        of_model, input_size=images.shape[-2:], cuda=torch.cuda.is_available())
    wrapped = {'images': images, 'flows': flow_gt, 'valids': valid}
    inputs  = io_adapter.prepare_inputs(inputs=wrapped)
    images  = images.to(device)
    I1, I2  = images[:, 0], images[:, 1]

    # ── clean predictions ────────────────────────────────────────────────────
    with torch.no_grad():
        clean_flow = of_model(inputs)['flows'].squeeze(0)

        raw_d = mde_model(inputs)
        _d = raw_d
        if _d.dim() == 2:   _d = _d.unsqueeze(0).unsqueeze(0)
        elif _d.dim() == 3: _d = _d.unsqueeze(1)
        proj_depth = disp_to_depth(_d.to(device).float())

        _logits = ss_model(inputs, return_logits=True)
        clean_ss = _logits.argmax(dim=1)
        road_mask = (_logits.argmax(dim=1) == 0).unsqueeze(1)
        road_mask = keep_largest_component(road_mask)

    # ── apply patch ──────────────────────────────────────────────────────────
    if patch_projection:
        planes = fit_plane_from_depth(proj_depth, K, road_mask)
        atk_I1, atk_I2, mask, _y, _x, _, _ = project_patch_on_scene(
            I1, I2, K, A=A,
            mde_model=mde_model, ss_model=ss_model,
            io_adapter=io_adapter, device=device,
            plane_aug=False,
            precomputed_depth=proj_depth,
            precomputed_road_mask=road_mask,
            precomputed_planes=planes,
        )
    else:
        atk_I1, atk_I2, mask, _y, _x = A(I1, I2)

    inputs_atk = replace_images_dic(
        inputs, torch.stack([atk_I1, atk_I2], dim=1).squeeze(0))

    # ── attacked predictions ─────────────────────────────────────────────────
    with torch.no_grad():
        atk_flow  = of_model(inputs_atk)['flows'].squeeze(0)
        atk_depth = mde_model(inputs_atk)
        atk_ss    = ss_model(inputs_atk)

    # ── render ───────────────────────────────────────────────────────────────
    d_c  = np_depth(raw_d)
    d_a  = np_depth(atk_depth)
    vmin = min(d_c.min(), d_a.min())
    vmax = max(d_c.max(), d_a.max())

    panels = [
        img_to_uint8(I1),                    img_to_uint8(atk_I1),
        flow_to_rgb(clean_flow),             flow_to_rgb(atk_flow),
        depth_to_rgb(d_c, vmin, vmax),       depth_to_rgb(d_a, vmin, vmax),
        seg_to_rgb(clean_ss),                seg_to_rgb(atk_ss),
    ]
    return panels, atk_I1


def save_figure(panels, idx, patch_path, patch_size, patch_projection, out_dir):
    titles = [
        'Clean image',
        'Attacked image  (diffusion patch)',
        'Clean optical flow',
        'Attacked optical flow  (↓ target)',
        'Clean depth',
        'Attacked depth  (far target)',
        'Clean segmentation',
        'Attacked segmentation',
    ]
    row_labels = ['Image', 'Optical flow', 'Depth (MDE)', 'Segmentation']
    proj_tag = ' + road projection' if patch_projection else ''

    fig, axes = plt.subplots(4, 2, figsize=(22, 14))
    for ax, img, title in zip(axes.flatten(), panels, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=11, fontweight='bold', pad=4)
        ax.axis('off')
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=12, labelpad=8)

    patch_name = os.path.basename(patch_path)
    fig.suptitle(
        f'Adversarial diffusion patch — KITTI-15 sample #{idx}  '
        f'(size {patch_size}px{proj_tag})  [{patch_name}]',
        fontsize=12, y=1.005,
    )
    plt.tight_layout(pad=0.6)

    out_png = os.path.join(out_dir, f'diffusion_viz_{idx:03d}.png')
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_png


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sample_idx', type=int, default=12)
    p.add_argument('--sweep', type=int, nargs='+', metavar='IDX',
                   help='Run on multiple samples instead of --sample_idx')
    p.add_argument('--patch', default=_DEFAULT_PATCH,
                   help='Path to pre-trained diffusion patch PNG')
    p.add_argument('--patch_size', type=int, default=200)
    p.add_argument('--patch_projection', action='store_true')
    p.add_argument('--model_name', default='raft')
    p.add_argument('--mde_model', default='depth-anything-v2')
    p.add_argument('--ss_model', default='pspnet_cityscapes')
    p.add_argument('--output_dir', default='experiments/diffusion_viz')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.isfile(args.patch):
        raise FileNotFoundError(
            f"Diffusion patch not found: {args.patch}\n"
            f"Available patches:\n" +
            '\n'.join(
                f"  {f}" for f in sorted(
                    __import__('glob').glob(
                        os.path.join(_ROOT, 'experiments',
                                     'quick_physical_3task_diffusion', '*.png')))
            )
        )

    print(f"Patch: {args.patch}")
    print(f"Device: {device}")

    # dataset
    cfg = load_dataset_args('Kitti15')
    cfg['frames'] = 2
    cfg['split']  = 'training'
    full_ds = KITTI(**cfg)
    print(f"KITTI-15: {len(full_ds)} samples")

    # models (load once, reuse across samples)
    print("Loading models…")
    of_model, mde_model, ss_model = load_models(
        args.model_name, args.mde_model, args.ss_model, device)

    indices = args.sweep if args.sweep else [args.sample_idx % len(full_ds)]

    for idx in indices:
        idx = idx % len(full_ds)
        print(f"Sample {idx}…")
        panels, _ = run_one(
            idx, full_ds, of_model, mde_model, ss_model,
            args.patch, args.patch_size, args.patch_projection, device,
        )
        out = save_figure(panels, idx, args.patch, args.patch_size,
                          args.patch_projection, args.output_dir)
        print(f"  → {out}")

    print("Done.")


if __name__ == '__main__':
    main()
