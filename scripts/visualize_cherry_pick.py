#!/usr/bin/env python3
"""
Train an adversarial patch on a single KITTI-15 image pair (cherry-pick) and
produce a 4×2 visualization: image / flow / depth / segmentation, clean vs attacked.

Flow target: down.  MDE target: far.  SS target: targeted.

Usage:
  python scripts/visualize_cherry_pick.py --sample_idx 0 --steps 500 --patch_size 200
  python scripts/visualize_cherry_pick.py --sample_idx 12 --steps 300 --patch_projection
"""

import argparse
import os
import sys
from types import SimpleNamespace

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'flow_library'))  # for flow_errors dep

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

import ptlflow
import ptlflow.utils.io_adapter as io_adapter_lib

from datasets_utils.datasets import KITTI
from datasets_utils.dataset_utils import load_dataset_args
from models.model_utils import load_mde_model, load_seg_model
from utils.process_images import _to_numpy_image, _to_numpy_flow, replace_images_dic
from utils.seed import set_seed
from attacks.patch_attack import train_patch_ptlflow
from attacks.DetectionDefenses.helper_functions.patch_adversary import PatchAdversary
from attacks.patch_projection import (
    project_patch_on_scene, fit_plane_from_depth, keep_largest_component,
)
from attacks.adversarial_manhole.adv_manhole.texture_mapping.depth_utils import disp_to_depth
from metrics.attack_metrics import AttackMetricsTracker
from flow_library.flow_plot import colorplot_light


# Cityscapes 19-class RGB palette
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


# ── rendering helpers ─────────────────────────────────────────────────────────

def flow_to_rgb(flow_tensor, max_scale=None):
    if torch.is_tensor(flow_tensor):
        flow = flow_tensor.detach().cpu().float().numpy()
    else:
        flow = np.asarray(flow_tensor, dtype=np.float32)
    if flow.ndim == 4:
        flow = flow[0]
    flow_hw2 = np.nan_to_num(np.transpose(flow, (1, 2, 0)), nan=0.0)
    # Use auto_scale=True per panel so low-magnitude ego-motion fills the color
    # range rather than being washed out by a few high-magnitude outlier pixels.
    if max_scale is None:
        return colorplot_light(flow_hw2, auto_scale=True, return_max=False).astype(np.uint8)
    return colorplot_light(flow_hw2, auto_scale=False, max_scale=max_scale, return_max=False).astype(np.uint8)


def depth_to_rgb(depth_np, vmin, vmax):
    d = np.nan_to_num(depth_np, nan=0.0, posinf=0.0, neginf=0.0)
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


def img_to_uint8(tensor):
    return (_to_numpy_image(tensor) * 255).clip(0, 255).astype(np.uint8)


def _np_depth(t):
    d = t.detach().cpu().float().numpy() if torch.is_tensor(t) else np.asarray(t, dtype=np.float32)
    while d.ndim > 2:
        d = d[0]
    return np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)


# ── args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sample_idx', type=int, default=0,
                   help='Index into KITTI-15 training set')
    p.add_argument('--model_name', default='raft')
    p.add_argument('--mde_model', default='depth-anything-v2')
    p.add_argument('--ss_model', default='pspnet_cityscapes')
    p.add_argument('--patch_size', type=int, default=200)
    p.add_argument('--steps', type=int, default=300,
                   help='Gradient steps on the single image pair')
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--output_dir', default='experiments/cherry_pick')
    p.add_argument('--patch_projection', action='store_true',
                   help='Project patch onto the road plane')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--flow_weight', type=float, default=2.0)
    p.add_argument('--mde_weight', type=float, default=0.1)
    p.add_argument('--ss_weight', type=float, default=1.0)
    # diffusion patch
    p.add_argument('--diffusion', action='store_true',
                   help='Use diffusion (latent) patch parametrization instead of pixel')
    p.add_argument('--diffusion_model', default='sdxl',
                   choices=['sd14', 'sd15', 'sd21', 'sdxl'])
    p.add_argument('--diffusion_model_path', default='')
    p.add_argument('--diffusion_prompt', default='')
    p.add_argument('--diffusion_source_steps', type=int, default=50)
    p.add_argument('--diffusion_reverse_steps', type=int, default=25)
    p.add_argument('--diffusion_latent_eps', type=float, default=0.5)
    p.add_argument('--diffusion_seed', type=int, default=42)
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | sample_idx: {args.sample_idx}")

    # ── dataset: single sample ────────────────────────────────────────────────
    dataset_cfg = load_dataset_args('Kitti15')
    dataset_cfg['frames'] = 2
    dataset_cfg['split'] = 'training'
    full_ds = KITTI(**dataset_cfg)
    n_ds = len(full_ds)
    print(f"KITTI-15 training set: {n_ds} samples")
    idx = args.sample_idx % n_ds
    loader = DataLoader(Subset(full_ds, [idx]), batch_size=1, shuffle=False)
    loader.image_size = (full_ds.image_x_dim, full_ds.image_y_dim)

    # ── models ────────────────────────────────────────────────────────────────
    print("Loading models…")
    of_model = ptlflow.get_model(args.model_name, 'chairs').to(device)
    of_model.eval()
    for p_ in of_model.parameters():
        p_.requires_grad_(False)

    mde_model = load_mde_model(model_name=args.mde_model, device=device)
    ss_model = load_seg_model(model_name=args.ss_model, device=device)
    for p_ in ss_model.parameters():
        p_.requires_grad_(False)

    io_adapter = ptlflow.utils.io_adapter.IOAdapter(
        of_model, input_size=loader.image_size, cuda=torch.cuda.is_available())

    # ── train_patch_ptlflow args namespace ────────────────────────────────────
    train_args = SimpleNamespace(
        # flow target (down with directional hinge)
        target='down',
        down_loss='hinge',
        flow_target_magnitude=1.0,
        flow_target_angle_deg=270.0,
        loss=['hinge'],
        effective_flow_loss='directional_hinge',
        down_hinge_magnitude_weight=2.0,
        down_hinge_min_mag_ratio=1.0,
        down_hinge_horizontal_weight=0.1,
        down_hinge_vertical_weight=1.0,
        # MDE
        attack_mde=True,
        mde_model=args.mde_model,
        mde_target='far',
        mde_near_margin=0.1,
        # segmentation
        attack_ss=True,
        ss_model=args.ss_model,
        ss_target='targeted',
        ss_focal_gamma=2.0,
        # multi-task weights
        loss_weights=[args.flow_weight, args.mde_weight, args.ss_weight],
        weight_strategy='normalized',
        minmax_alpha_w=0.03,
        minmax_gamma=5.0,
        # patch
        patch_size=args.patch_size,
        patch_parametrization='diffusion' if args.diffusion else 'pixel',
        change_of_variables=True,
        random_loc=True,
        y_scale=3.0,
        # EOT
        eot_n=2,
        eot_angle=15.0,
        eot_scale_min=0.9,
        eot_scale_max=1.1,
        eot_color_jitter=0.1,
        eot_noise_std=0.01,
        # optimiser
        optimizer='adam',
        lr=args.lr,
        max_delta=0.008,
        n=1,             # 1 outer epoch (iterate once over the single sample)
        steps=args.steps,  # inner gradient steps per batch
        # diffusion patch params (ignored when patch_parametrization='pixel')
        diffusion_model=args.diffusion_model,
        diffusion_model_path=args.diffusion_model_path,
        diffusion_prompt=args.diffusion_prompt,
        diffusion_init_mode='random',
        diffusion_base_image='',
        diffusion_dtype='auto',
        diffusion_decode_mode='denoise',
        diffusion_source_steps=args.diffusion_source_steps,
        diffusion_reverse_steps=args.diffusion_reverse_steps,
        diffusion_guidance_scale=7.5,
        diffusion_latent_eps=args.diffusion_latent_eps,
        diffusion_null_inner_steps=15,
        diffusion_null_epsilon=1e-5,
        diffusion_optimizer='adam',
        diffusion_seed=args.diffusion_seed,
        # projection / flow shift
        patch_projection=args.patch_projection,
        plane_aug=False,
        flow_shift=False,
        flow_shift_mode='fixed',
        flow_shift_scale=1.0,
        flow_shift_max=80.0,
        # regularisation
        tv_weight=0.0,
        nps_weight=0.0,
        # defense
        defense='none',
        # dataset (used internally for the depth-target percentile loader)
        dataset='Kitti15',
        model_name=args.model_name,
        attack_type='patch',
        output_dir=args.output_dir,
        experiment_name='cherry_pick',
        small_run=False,
        subset_size=1,
        # logging
        save_artifacts=False,
        save_diploma_artifacts=False,
        save_patch_every=0,
        saved_iterations=[],
        baseline=False,
        trained_patch='',
    )

    # ── train ─────────────────────────────────────────────────────────────────
    print(f"Training patch on sample {idx}: {args.steps} gradient steps…")
    tracker = AttackMetricsTracker(
        output_dir=args.output_dir,
        experiment_name='cherry_pick',
        args=train_args,
        train=True,
    )
    raw_patch = train_patch_ptlflow(
        train_args, of_model, loader, device, io_adapter,
        tracker, mde_model, ss_model,
    )

    # freeze the trained patch (no more random transforms)
    patch_tensor = raw_patch.get_P(Mask=True).detach().cpu()
    A = PatchAdversary(
        patch_tensor,
        size=args.patch_size,
        angle=0, scale=1,
        change_of_variable=False,
        random_location=False,
        image_size=loader.image_size,
        ellipse_scale_y=1.0,
    ).to(device)
    tracker.finalize()

    # ── clean + attacked inference ────────────────────────────────────────────
    print("Running inference for visualization…")
    for images, flow_gt, valid, meta, K in loader:
        break

    io_adapter = ptlflow.utils.io_adapter.IOAdapter(
        of_model, input_size=images.shape[-2:], cuda=torch.cuda.is_available())
    wrapped = {'images': images, 'flows': flow_gt, 'valids': valid}
    inputs = io_adapter.prepare_inputs(inputs=wrapped)
    images = images.to(device)
    I1, I2 = images[:, 0], images[:, 1]

    with torch.no_grad():
        clean_flow = of_model(inputs)['flows'].squeeze(0)

        raw_depth = mde_model(inputs)
        clean_depth = raw_depth
        _d = raw_depth
        if _d.dim() == 2:
            _d = _d.unsqueeze(0).unsqueeze(0)
        elif _d.dim() == 3:
            _d = _d.unsqueeze(1)
        proj_depth = disp_to_depth(_d.to(device).float())

        _ss_logits = ss_model(inputs, return_logits=True)
        clean_ss = _ss_logits.argmax(dim=1)
        proj_road_mask = (_ss_logits.argmax(dim=1) == 0).unsqueeze(1)
        proj_road_mask = keep_largest_component(proj_road_mask)

    if args.patch_projection:
        proj_planes = fit_plane_from_depth(proj_depth, K, proj_road_mask)
        atk_I1, atk_I2, mask, _y, _x, _, _ = project_patch_on_scene(
            I1, I2, K, A=A,
            mde_model=mde_model, ss_model=ss_model,
            io_adapter=io_adapter, device=device,
            plane_aug=False,
            precomputed_depth=proj_depth,
            precomputed_road_mask=proj_road_mask,
            precomputed_planes=proj_planes,
        )
    else:
        atk_I1, atk_I2, mask, _y, _x = A(I1, I2)

    inputs_atk = replace_images_dic(inputs, torch.stack([atk_I1, atk_I2], dim=1).squeeze(0))

    with torch.no_grad():
        atk_flow  = of_model(inputs_atk)['flows'].squeeze(0)
        atk_depth = mde_model(inputs_atk)
        atk_ss    = ss_model(inputs_atk)

    # ── render panels ─────────────────────────────────────────────────────────
    # Flow: auto-scale per panel so ego-motion fills the color range instead of
    # being washed out relative to the high-magnitude car blob.
    d_clean = _np_depth(clean_depth)
    d_atk   = _np_depth(atk_depth)
    d_vmin  = min(d_clean.min(), d_atk.min())
    d_vmax  = max(d_clean.max(), d_atk.max())

    panels = [
        img_to_uint8(I1),                       img_to_uint8(atk_I1),
        flow_to_rgb(clean_flow),                 flow_to_rgb(atk_flow),
        depth_to_rgb(d_clean, d_vmin, d_vmax),  depth_to_rgb(d_atk, d_vmin, d_vmax),
        seg_to_rgb(clean_ss),                   seg_to_rgb(atk_ss),
    ]
    titles = [
        'Clean image',
        f'Attacked image  ({patch_tag} patch, {args.patch_size}px)',
        'Clean optical flow',
        'Attacked optical flow  (↓ target)',
        'Clean depth',
        'Attacked depth  (far target)',
        'Clean segmentation',
        'Attacked segmentation',
    ]
    row_labels = ['Image', 'Optical flow', 'Depth (MDE)', 'Segmentation']

    fig, axes = plt.subplots(4, 2, figsize=(22, 14))
    for i, (ax, img, title) in enumerate(zip(axes.flatten(), panels, titles)):
        ax.imshow(img)
        ax.set_title(title, fontsize=11, fontweight='bold', pad=4)
        ax.axis('off')
    # row annotations on the left
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=12, rotation=90, labelpad=8)
        axes[row, 0].yaxis.label.set_visible(True)

    patch_tag = 'diffusion' if args.diffusion else 'pixel'
    proj_tag  = ' + road projection' if args.patch_projection else ''
    fig.suptitle(
        f'Adversarial patch ({patch_tag}) — KITTI-15 sample #{idx}  '
        f'({args.steps} steps, size {args.patch_size}px{proj_tag})',
        fontsize=13, y=1.005,
    )
    plt.tight_layout(pad=0.6)

    out_png = os.path.join(args.output_dir, f'cherry_pick_{idx:03d}.png')
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Visualization saved → {out_png}")

    patch_png = os.path.join(args.output_dir, f'patch_{idx:03d}.png')
    A.save_png(patch_png)
    print(f"Patch saved       → {patch_png}")


if __name__ == '__main__':
    main()
