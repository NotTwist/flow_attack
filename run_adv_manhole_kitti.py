"""
Run adversarial-manhole attack adapted for KITTI-15 dataset.

Trains a physical adversarial patch using adv-manhole's framework on KITTI,
then evaluates it with unified metrics (optical flow AEE, depth RMSE, segmentation ASR)
for a direct comparison against the main patch attack.

Usage:
    python run_adv_manhole_kitti.py [options]

Key options:
    --mde_model         MDE backbone (default: depth-anything-v2)
    --ss_model          Segmentation backbone (default: segformer_cityscapes)
    --flow_model        Optical flow model for evaluation (default: raft)
    --epochs            Training epochs (default: 25)
    --batch_size        Training batch size (default: 4)
    --output_dir        Output directory (default: experiment_data/adv_manhole_kitti)
    --trained_patch     Path to pre-trained patch PNG (skips training if provided)
    --small_run         Use only 32 KITTI samples (for debugging)
    --subset_size       Limit KITTI to N samples
"""

import os
import sys
import argparse
import math
import numpy as np
from argparse import Namespace
from PIL import Image

# ---- adv_manhole must be importable before other imports ----
_HERE = os.path.dirname(os.path.abspath(__file__))
_ADV_MANHOLE_DIR = os.path.join(_HERE, 'attacks', 'adversarial_manhole')
if _ADV_MANHOLE_DIR not in sys.path:
    sys.path.insert(0, _ADV_MANHOLE_DIR)

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm

import ptlflow
import ptlflow.utils.io_adapter as io_adapter_lib

from datasets_utils.datasets import KITTI
from datasets_utils.dataset_utils import load_dataset_args, prepare_dataloader
from models.model_utils import load_mde_model, load_seg_model
from metrics.attack_metrics import AttackMetricsTracker
from utils.seed import set_seed

from adv_manhole.attack.framework import AdvManholeFramework
from adv_manhole.attack.losses import AdvManholeLosses
from adv_manhole.attack.naturalness import AdvContentLoss
from adv_manhole.texture_mapping.depth_mapping import DepthTextureMapping
from adv_manhole.texture_mapping.depth_utils import depth_to_local_coordinates

from attacks.adv_manhole_kitti.model_adapters import MDEModelAdapter, SSModelAdapter
from attacks.adv_manhole_kitti.kitti_data_adapter import (
    KITTIAdvManholeDataset,
    make_adv_manhole_dataset_dict,
)


def _load_manhole_set(manhole_dir, image_size=256):
    """Load reference manhole images from a directory as a (N,3,H,W) tensor."""
    loader = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    images = []
    for fname in sorted(os.listdir(manhole_dir)):
        fpath = os.path.join(manhole_dir, fname)
        if not os.path.isfile(fpath):
            continue
        try:
            img = Image.open(fpath).convert('RGB')
            images.append(loader(img).unsqueeze(0))
        except Exception:
            pass
    if not images:
        raise FileNotFoundError(f'No images found in {manhole_dir}')
    return torch.cat(images, dim=0)  # (N,3,H,W)


def parse_args():
    parser = argparse.ArgumentParser(description='adversarial-manhole on KITTI')
    parser.add_argument('--mde_model', default='depth-anything-v2')
    parser.add_argument('--ss_model', default='segformer_cityscapes')
    parser.add_argument('--flow_model', default='raft')
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--train_fraction', type=float, default=0.8,
                        help='Fraction of KITTI training set used for patch training')
    parser.add_argument('--tex_scale', type=float, default=100.0,
                        help='Patch footprint in cm (100 = 1m x 1m manhole)')
    parser.add_argument('--eval_shift_x', type=float, default=1500.0,
                        help='Deterministic eval patch depth offset in cm')
    parser.add_argument('--eval_shift_y', type=float, default=None,
                        help='Deterministic eval patch lateral offset in cm; default centers patch around y=0')
    parser.add_argument('--eval_auto_place', action='store_true', default=True,
                        help='Place eval patch from surface-coordinate quantiles instead of fixed offsets')
    parser.add_argument('--eval_target_coverage', type=float, default=0.015,
                        help='Approximate visible eval patch area as fraction of image when auto-placing.')
    parser.add_argument('--eval_min_coverage', type=float, default=0.001,
                        help='Warn when eval patch mask coverage is below this fraction.')
    parser.add_argument('--train_target_coverage', type=float, default=0.015,
                        help='Approximate visible train patch area as fraction of image when robust KITTI placement is enabled.')
    parser.add_argument('--disable_train_auto_place', action='store_true',
                        help='Use original adv-manhole random surface-coordinate shifts during training.')
    parser.add_argument('--output_dir', default='experiment_data/adv_manhole_kitti')
    parser.add_argument('--experiment_name', default='adv_manhole_kitti_comparison')
    parser.add_argument('--save_artifacts', action='store_true',
                        help='Save eval images/flows/masks for qualitative inspection.')
    parser.add_argument('--eval_artifact_limit', type=int, default=0,
                        help='Maximum number of eval batches to save artifacts for. 0 = no limit when --save_artifacts is enabled.')
    parser.add_argument('--trained_patch', default='',
                        help='Path to a pre-trained patch PNG; skips training if provided')
    parser.add_argument('--small_run', action='store_true',
                        help='Use only 32 KITTI samples for fast debugging')
    parser.add_argument('--subset_size', type=int, default=0,
                        help='Limit dataset to N samples (0 = all)')
    parser.add_argument('--eval_mode', choices=['training', 'testing'], default='testing',
                        help='KITTI15 split used for unified evaluation')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--road_class_idx', type=int, default=0,
                        help='Cityscapes class index for road (0)')
    return parser.parse_args()


def _load_flow_model(model_name, device):
    model_ref = ptlflow.get_model_reference(model_name)
    ckpt = next(iter(model_ref.pretrained_checkpoints))
    model = ptlflow.get_model(model_name, ckpt)
    return model.to(device).eval()


def _compute_flow(flow_model, images, device):
    """
    Run ptlflow model on an (N,2,3,H,W) image pair tensor.
    Returns flow (N,2,H,W).
    """
    io_adapter = io_adapter_lib.IOAdapter(
        flow_model, input_size=images.shape[-2:], cuda=device.type == 'cuda'
    )
    wrapped = {'images': images, 'flows': None, 'valids': None}
    inputs = io_adapter.prepare_inputs(inputs=wrapped)
    with torch.no_grad():
        pred = flow_model(inputs)['flows']
    # pred shape: (N,1,2,H,W) or (N,2,H,W)
    if pred.ndim == 5:
        pred = pred.squeeze(1)
    return pred  # (N,2,H,W)


def _unwrap_k_dict(K_dict):
    """Convert DataLoader-collated K_dict (tensor values) back to Python scalars."""
    result = {}
    for k, v in K_dict.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.squeeze().item() if v.numel() == 1 else v.squeeze().tolist()
        else:
            result[k] = v
    return result


def _camera_config_for_shape(K_dict, height, width):
    """Make adv-manhole camera config dimensions match the actual tensor shape."""
    K_dict = dict(_unwrap_k_dict(K_dict))
    old_width = float(K_dict.get("image_width", width) or width)
    old_fx = float(K_dict.get("fx", 0.0) or 0.0)

    K_dict["image_width"] = int(width)
    K_dict["image_height"] = int(height)

    if old_fx > 0.0 and old_width > 0.0:
        fx_scaled = old_fx * (float(width) / old_width)
        K_dict["fx"] = fx_scaled
        K_dict["fov"] = 2.0 * math.degrees(math.atan(float(width) / (2.0 * fx_scaled)))

    return K_dict


def _surface_coors_for_frame(frame, mde_model, K_dict):
    """
    Compute local_surface_coors (3,H,W) for a single (3,H,W) frame tensor.
    K_dict may have tensor values (from DataLoader collation) — unwrapped here.
    """
    with torch.no_grad():
        depth = mde_model({'images': frame.unsqueeze(0).unsqueeze(0)})  # (1,1,H,W) meters
    depth_np = depth.squeeze().cpu().numpy()  # (H,W)
    K_dict = _camera_config_for_shape(K_dict, depth_np.shape[0], depth_np.shape[1])
    surface_xyz = depth_to_local_coordinates(depth_np / 1000.0, K_dict)  # (H,W,3) cm
    return torch.from_numpy(surface_xyz).permute(2, 0, 1).float()  # (3,H,W)


def _auto_eval_offsets(surface_coors, tex_scale, device):
    """
    Choose deterministic patch offsets from the lower image half.

    adv-manhole's mapper interprets offsets as the min corner of a square in
    surface-coordinate space. KITTI depth/relative-MDE scale varies per frame,
    so fixed centimeter offsets can miss the rendered surface entirely.
    """
    offsets = []
    for sc in surface_coors:
        _, h, _ = sc.shape
        roi = sc[:, h // 2 :, :].reshape(3, -1)
        finite = torch.isfinite(roi).all(dim=0)
        if finite.sum() < 16:
            roi = sc.reshape(3, -1)
            finite = torch.isfinite(roi).all(dim=0)
        if finite.sum() < 16:
            offsets.append([0.0, -0.5 * tex_scale, 0.0])
            continue

        x_vals = roi[0, finite]
        y_vals = roi[1, finite]
        x_min = torch.quantile(x_vals, 0.55).item() - 0.5 * tex_scale
        y_min = torch.quantile(y_vals, 0.50).item() - 0.5 * tex_scale
        offsets.append([max(0.0, x_min), y_min, 0.0])

    return torch.tensor(offsets, device=device, dtype=torch.float32)


def _auto_eval_footprints_from_image_roi(
    surface_coors,
    *,
    target_coverage,
    fallback_tex_scale,
    device,
):
    """
    Choose deterministic eval footprints from a lower-center image ROI.

    The original adv-manhole placement samples a square in local 3D surface
    coordinates. On KITTI, those coordinates are reconstructed from relative
    MDE predictions, so a fixed 100 cm square can easily cover almost no image
    pixels. This helper inverts the choice: first select a visible image ROI,
    then choose the smallest square in surface-coordinate x/y that covers the
    ROI's finite surface points. The texture mapping remains adv-manhole's
    original mapping; only the deterministic eval placement is made robust.
    """
    target_coverage = max(float(target_coverage), 1e-6)
    side_frac = min(0.8, max(0.03, target_coverage ** 0.5))

    offsets = []
    scales = []
    for sc in surface_coors:
        _, h, w = sc.shape
        roi_h = max(8, int(round(h * side_frac)))
        roi_w = max(8, int(round(w * side_frac)))
        cy = int(round(h * 0.72))
        cx = w // 2
        y0 = max(0, min(h - roi_h, cy - roi_h // 2))
        x0 = max(0, min(w - roi_w, cx - roi_w // 2))

        roi = sc[:, y0 : y0 + roi_h, x0 : x0 + roi_w].reshape(3, -1)
        finite = torch.isfinite(roi).all(dim=0)

        # If the chosen ROI has invalid geometry, fall back to the lower half.
        if finite.sum() < 16:
            roi = sc[:, h // 2 :, :].reshape(3, -1)
            finite = torch.isfinite(roi).all(dim=0)

        if finite.sum() < 16:
            offsets.append([0.0, -0.5 * fallback_tex_scale, 0.0])
            scales.append(float(fallback_tex_scale))
            continue

        x_vals = roi[0, finite]
        y_vals = roi[1, finite]
        x_lo = torch.quantile(x_vals, 0.05)
        x_hi = torch.quantile(x_vals, 0.95)
        y_lo = torch.quantile(y_vals, 0.05)
        y_hi = torch.quantile(y_vals, 0.95)

        x_span = (x_hi - x_lo).abs().clamp_min(1e-3)
        y_span = (y_hi - y_lo).abs().clamp_min(1e-3)
        scale = torch.maximum(x_span, y_span)
        scale = torch.nan_to_num(scale, nan=float(fallback_tex_scale), posinf=float(fallback_tex_scale))
        scale = scale.clamp_min(1e-3)

        x_mid = 0.5 * (x_lo + x_hi)
        y_mid = 0.5 * (y_lo + y_hi)
        x_min = x_mid - 0.5 * scale
        y_min = y_mid - 0.5 * scale

        offsets.append([float(x_min.item()), float(y_min.item()), 0.0])
        scales.append(float(scale.item()))

    return (
        torch.tensor(offsets, device=device, dtype=torch.float32),
        torch.tensor(scales, device=device, dtype=torch.float32),
    )


class KITTIRobustDepthTextureMapping(DepthTextureMapping):
    """DepthTextureMapping with robust KITTI placement for relative-depth coords.

    The original adv-manhole random offsets are specified in surface-coordinate
    units. For KITTI we reconstruct those coordinates from relative MDE, so fixed
    centimeter-like ranges often place the patch outside the visible image. This
    subclass keeps the original texture mapping, but chooses training footprints
    from a visible lower-image ROI whenever explicit offsets are not provided.
    """

    def __init__(self, *args, auto_place=True, target_coverage=0.015, **kwargs):
        super().__init__(*args, **kwargs)
        self.auto_place = bool(auto_place)
        self.target_coverage = float(target_coverage)

    def __call__(
        self,
        texture,
        surface_xyz,
        background,
        batch_size,
        tex_scales=None,
        xyz_offsets=None,
        random_scale=True,
        random_shift=True,
    ):
        if self.auto_place and tex_scales is None and xyz_offsets is None:
            xyz_offsets, tex_scales = _auto_eval_footprints_from_image_roi(
                surface_xyz,
                target_coverage=self.target_coverage,
                fallback_tex_scale=self.tex_scale,
                device=self.device,
            )
            random_scale = False
            random_shift = False
        return super().__call__(
            texture,
            surface_xyz,
            background,
            batch_size,
            tex_scales=tex_scales,
            xyz_offsets=xyz_offsets,
            random_scale=random_scale,
            random_shift=random_shift,
        )


def main():
    args = parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    set_seed(42)
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Load backbone models                                              #
    # ------------------------------------------------------------------ #
    print('Loading MDE model...')
    mde_model = load_mde_model(model_name=args.mde_model, device=device)

    print('Loading segmentation model...')
    ss_model = load_seg_model(model_name=args.ss_model, device=device)

    # ------------------------------------------------------------------ #
    # 2. Build adv-manhole model adapters                                  #
    # ------------------------------------------------------------------ #
    mde_wrapped = MDEModelAdapter(mde_model, device)
    ss_wrapped = SSModelAdapter(ss_model, device)

    # ------------------------------------------------------------------ #
    # 3. Build DepthTextureMapping tuned for KITTI scene scale             #
    # ------------------------------------------------------------------ #
    # tex_scale is in cm; KITTI road spans ~500-5000 cm in depth.
    # Default 100 cm = 1m × 1m patch (manhole-sized).
    # random_shift_x: place patch at 5-25m depth in front of camera.
    # random_shift_y: ±2m lateral offset to cover road width.
    depth_planar_mapping = KITTIRobustDepthTextureMapping(
        texture_res=256,
        tex_scale=args.tex_scale,
        tex_offset=[0.0, 0.0],
        random_scale=(0.0, args.tex_scale * 0.1),
        random_shift_x=(500.0, 2500.0),
        random_shift_y=(-200.0, 200.0),
        with_circle_mask=True,
        device=device,
        auto_place=not args.disable_train_auto_place,
        target_coverage=args.train_target_coverage,
    )

    # ------------------------------------------------------------------ #
    # 4. Patch texture                                                     #
    # ------------------------------------------------------------------ #
    patch_texture_var = nn.Parameter(
        torch.rand(3, 256, 256, device=device), requires_grad=True
    )

    # ------------------------------------------------------------------ #
    # 5. Losses                                                            #
    # ------------------------------------------------------------------ #
    manhole_dir = os.path.join(_ADV_MANHOLE_DIR, 'adversarial_example')
    # load_manhole_set returns (N,3,256,256) tensor of reference manhole images
    candidate_images = _load_manhole_set(manhole_dir, image_size=256).to(device)

    adv_content_loss = AdvContentLoss(candidate_images=candidate_images)

    adversarial_losses = AdvManholeLosses(
        adv_content_loss=adv_content_loss,
        mde_loss_weight=2.0,
        ss_ua_loss_weight=0.5,
        ss_ta_loss_weight=0.5,
        tv_loss_weight=1.0,
        content_loss_weight=0.5,
        background_loss_weight=0.0,
    )

    # ------------------------------------------------------------------ #
    # 6. Augmentations                                                     #
    # ------------------------------------------------------------------ #
    texture_aug = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ])
    output_aug = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.1),
    ])

    # ------------------------------------------------------------------ #
    # 7. Build KITTI dataset in adv-manhole format                         #
    # ------------------------------------------------------------------ #
    if args.trained_patch == '':
        print('Building KITTI dataset for adv-manhole training...')
        dataset_args = load_dataset_args('Kitti15')
        dataset_args['split'] = 'training'
        dataset_args['has_gt'] = True
        dataset_args['has_depth'] = True
        kitti_base = KITTI(**dataset_args)

        n_total = len(kitti_base)
        if args.subset_size > 0:
            kitti_base = torch.utils.data.Subset(kitti_base, range(min(args.subset_size, n_total)))
        elif args.small_run:
            kitti_base = torch.utils.data.Subset(kitti_base, range(min(32, n_total)))

        adv_dataset = make_adv_manhole_dataset_dict(
            kitti_dataset=kitti_base,
            mde_model=mde_model,
            ss_model=ss_model,
            device=device,
            batch_size=args.batch_size,
            train_fraction=args.train_fraction,
            road_class_idx=args.road_class_idx,
        )
        n_train = len(adv_dataset['train'].dataset)
        n_val = len(adv_dataset['validation'].dataset)

        # ------------------------------------------------------------------ #
        # 8. Build framework and train                                         #
        # ------------------------------------------------------------------ #
        optimizer = optim.Adam([patch_texture_var], lr=args.lr)

        framework = AdvManholeFramework(
            optimizer=optimizer,
            mde_model=mde_wrapped,
            ss_model=ss_wrapped,
            loss=adversarial_losses,
            patch_texture_var=patch_texture_var,
            depth_planar_mapping=depth_planar_mapping,
            texture_augmentation=texture_aug,
            output_augmentation=output_aug,
            device=device,
        )

        print(f'Training adv-manhole patch on KITTI ({n_train} train / {n_val} val samples)...')
        framework.train(
            epochs=args.epochs,
            dataset=adv_dataset,
            train_total_batch=n_train // args.batch_size + (1 if n_train % args.batch_size else 0),
            val_total_batch=n_val // args.batch_size + (1 if n_val % args.batch_size else 0),
            log_prediction_every=5,
            log_name='adv_manhole_kitti',
        )

        # Save trained patch
        patch_path = os.path.join(args.output_dir, 'trained_patch_kitti.png')
        if not torch.isfinite(patch_texture_var).all():
            print('Warning: trained patch contained NaN/Inf; replacing non-finite values before saving.')
            patch_texture_var.data = torch.nan_to_num(
                patch_texture_var.data, nan=0.0, posinf=1.0, neginf=0.0
            ).clamp_(0.0, 1.0)
        patch_np = (patch_texture_var.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(patch_np).save(patch_path)
        print(f'Patch saved to {patch_path}')
        trained_patch = patch_texture_var.detach()
    else:
        # Load pre-trained patch
        print(f'Loading pre-trained patch from {args.trained_patch}...')
        patch_img = transforms.ToTensor()(Image.open(args.trained_patch)).to(device)
        trained_patch = patch_img  # (3,256,256) float [0,1]

    # ------------------------------------------------------------------ #
    # 9. Unified evaluation with optical flow metrics                      #
    # ------------------------------------------------------------------ #
    print('Loading optical flow model for evaluation...')
    flow_model = _load_flow_model(args.flow_model, device)
    for param in flow_model.parameters():
        param.requires_grad = False

    # Minimal args namespace for AttackMetricsTracker
    tracker_args = Namespace(
        model_name=args.flow_model,
        attack_type='adv_manhole',
        dataset='Kitti15',
        saved_iterations=[],
        mde_model=args.mde_model,
        ss_model=args.ss_model,
    )

    eval_loader, has_gt = prepare_dataloader(
        mode=args.eval_mode,
        dataset_name='Kitti15',
        batch_size=1,
        small_run=args.small_run,
        subset_size=args.subset_size,
        has_depth=False,
    )

    eval_tracker = AttackMetricsTracker(
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        run_name=f'adv_manhole_kitti_eval',
        args=tracker_args,
    )

    print('Evaluating trained patch...')
    mask_coverages = []
    for batch_idx, (images, flow_gt, valid, disp, K_dict) in enumerate(tqdm(eval_loader)):
        images = images.to(device)   # (1,2,3,H,W) float [0,1]
        I1 = images[:, 0]            # (1,3,H,W)
        I2 = images[:, 1]            # (1,3,H,W)

        # Compute surface coordinates for both frames.
        # K_dict has tensor values after DataLoader collation; _surface_coors_for_frame unwraps them.
        sc_I1 = _surface_coors_for_frame(I1[0], mde_model, K_dict).unsqueeze(0).to(device)  # (1,3,H,W)
        sc_I2 = _surface_coors_for_frame(I2[0], mde_model, K_dict).unsqueeze(0).to(device)  # (1,3,H,W)

        # Deterministic patch placement (same tex_scales/xyz_offsets for I1 and I2)
        batch_sz = I1.shape[0]
        if args.eval_auto_place:
            xyz_offsets, tex_scales = _auto_eval_footprints_from_image_roi(
                sc_I1,
                target_coverage=args.eval_target_coverage,
                fallback_tex_scale=args.tex_scale,
                device=device,
            )
        else:
            tex_scales = torch.tensor([args.tex_scale] * batch_sz, device=device)
            eval_shift_y = (
                depth_planar_mapping.tex_offset[1]
                if args.eval_shift_y is None
                else args.eval_shift_y
            )
            xyz_offsets = torch.tensor(
                [[args.eval_shift_x, eval_shift_y, 0.0]] * batch_sz,
                device=device,
            )

        batched_texture = trained_patch.unsqueeze(0).repeat(batch_sz, 1, 1, 1)

        with torch.no_grad():
            adv_I1, mask = depth_planar_mapping(
                batched_texture, sc_I1, I1, batch_sz,
                tex_scales=tex_scales, xyz_offsets=xyz_offsets,
                random_scale=False, random_shift=False,
            )
            mask_coverage = float(mask.float().mean().item())
            mask_coverages.append(mask_coverage)
            if batch_idx < 5 and mask_coverage < args.eval_min_coverage:
                print(
                    f'Warning: eval batch {batch_idx} has low patch mask coverage '
                    f'({mask_coverage:.6f}).'
                )
            adv_I2, _ = depth_planar_mapping(
                batched_texture, sc_I2, I2, batch_sz,
                tex_scales=tex_scales, xyz_offsets=xyz_offsets,
                random_scale=False, random_shift=False,
            )

            # Optical flow: clean pair
            clean_pair = images                                        # (1,2,3,H,W)
            adv_pair = torch.stack([adv_I1, adv_I2], dim=1)           # (1,2,3,H,W)

            original_flow = _compute_flow(flow_model, clean_pair, device)  # (1,2,H,W)
            attacked_flow = _compute_flow(flow_model, adv_pair, device)    # (1,2,H,W)

            # MDE: clean and attacked I1
            original_depth = mde_model({'images': images})             # (1,1,H,W)
            attacked_depth = mde_model({'images': adv_pair})           # (1,1,H,W)

            # Segmentation: clean and attacked I1
            original_ss = ss_model(I1)                                 # (1,H,W) class indices
            attacked_ss = ss_model(adv_I1)                             # (1,H,W) class indices

        eval_tracker.update(
            original_flow.squeeze(0),
            attacked_flow.squeeze(0),
            gt_flow=flow_gt.to(device).squeeze(0) if has_gt and flow_gt is not None else None,
            target_flow=None,
            inverse_flow=None,
            valid=valid,
            mask=mask.cpu(),
            original_depth=original_depth,
            attacked_depth=attacked_depth,
            target_depth=None,
            original_seg=original_ss,
            attacked_seg=attacked_ss,
            target_seg=None,
        )

        artifact_limit = int(getattr(args, 'eval_artifact_limit', 0) or 0)
        if args.save_artifacts and (artifact_limit <= 0 or batch_idx < artifact_limit):
            eval_tracker.save_artifact(I1, f'eval_{batch_idx:04d}_clean_image', artifact_type='image')
            eval_tracker.save_artifact(adv_I1, f'eval_{batch_idx:04d}_attacked_image', artifact_type='image')
            eval_tracker.save_artifact(mask, f'eval_{batch_idx:04d}_patch_mask', artifact_type='image')
            eval_tracker.save_artifact(original_flow.squeeze(0), f'eval_{batch_idx:04d}_clean_flow', artifact_type='flow')
            eval_tracker.save_artifact(attacked_flow.squeeze(0), f'eval_{batch_idx:04d}_attacked_flow', artifact_type='flow')
            eval_tracker.save_artifact(original_depth, f'eval_{batch_idx:04d}_clean_depth', artifact_type='depth')
            eval_tracker.save_artifact(attacked_depth, f'eval_{batch_idx:04d}_attacked_depth', artifact_type='depth')
            eval_tracker.save_artifact(original_ss, f'eval_{batch_idx:04d}_clean_ss', artifact_type='ss')
            eval_tracker.save_artifact(attacked_ss, f'eval_{batch_idx:04d}_attacked_ss', artifact_type='ss')

    eval_tracker.finalize()
    if mask_coverages:
        print(
            'Patch mask coverage: '
            f'min={min(mask_coverages):.6f}, '
            f'mean={float(np.mean(mask_coverages)):.6f}, '
            f'max={max(mask_coverages):.6f}'
        )
    print(f'Evaluation complete. Results logged to MLflow experiment: {args.experiment_name}')


if __name__ == '__main__':
    main()
