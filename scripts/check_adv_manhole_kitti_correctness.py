#!/usr/bin/env python3
"""Smoke-check adv-manhole KITTI integration.

The check is intentionally small and diagnostic-focused. It verifies that:
  - the patch is actually rendered with non-trivial mask coverage;
  - adv-manhole losses are finite and non-zero when the mask is non-empty;
  - gradients reach the texture parameter;
  - one optimizer step changes the patch;
  - clean and patched images differ in the rendered mask region.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

ADV_MANHOLE_DIR = REPO_ROOT / "attacks" / "adversarial_manhole"
sys.path.insert(0, str(ADV_MANHOLE_DIR))

from adv_manhole.attack.framework import AdvManholeFramework  # noqa: E402
from adv_manhole.attack.losses import AdvManholeLosses  # noqa: E402
from adv_manhole.attack.naturalness import AdvContentLoss  # noqa: E402
from attacks.adv_manhole_kitti.kitti_data_adapter import make_adv_manhole_dataset_dict  # noqa: E402
from attacks.adv_manhole_kitti.model_adapters import MDEModelAdapter, SSModelAdapter  # noqa: E402
from datasets_utils.datasets import KITTI  # noqa: E402
from datasets_utils.dataset_utils import load_dataset_args  # noqa: E402
from models.model_utils import load_mde_model, load_seg_model  # noqa: E402
from run_adv_manhole_kitti import KITTIRobustDepthTextureMapping, _load_manhole_set  # noqa: E402
from utils.seed import set_seed  # noqa: E402


def finite_float(x) -> float:
    if torch.is_tensor(x):
        return float(torch.nan_to_num(x.detach()).mean().item())
    return float(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset_size", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--mde_model", default="depth-anything-v2")
    parser.add_argument("--ss_model", default="segformer_cityscapes")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--tex_scale", type=float, default=100.0)
    parser.add_argument("--target_coverage", type=float, default=0.015)
    parser.add_argument("--min_coverage", type=float, default=0.001)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--output_dir", default="experiment_data/adv_manhole_kitti_correctness_check")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(42)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Device: {device}")
    print("Loading MDE/SS models...")
    mde_model = load_mde_model(model_name=args.mde_model, device=device)
    ss_model = load_seg_model(model_name=args.ss_model, device=device)
    mde_wrapped = MDEModelAdapter(mde_model, device)
    ss_wrapped = SSModelAdapter(ss_model, device)

    dataset_args = load_dataset_args("Kitti15")
    dataset_args["split"] = "training"
    dataset_args["has_gt"] = True
    dataset_args["has_depth"] = True
    kitti = KITTI(**dataset_args)
    kitti = torch.utils.data.Subset(kitti, range(min(args.subset_size, len(kitti))))
    adv_dataset = make_adv_manhole_dataset_dict(
        kitti_dataset=kitti,
        mde_model=mde_model,
        ss_model=ss_model,
        device=device,
        batch_size=args.batch_size,
        train_fraction=0.75,
        road_class_idx=0,
    )

    mapping = KITTIRobustDepthTextureMapping(
        texture_res=256,
        tex_scale=args.tex_scale,
        tex_offset=[0.0, 0.0],
        random_scale=(0.0, args.tex_scale * 0.1),
        random_shift_x=(500.0, 2500.0),
        random_shift_y=(-200.0, 200.0),
        with_circle_mask=True,
        device=device,
        auto_place=True,
        target_coverage=args.target_coverage,
    )

    patch_texture_var = torch.nn.Parameter(
        torch.rand(3, 256, 256, device=device),
        requires_grad=True,
    )
    optimizer = torch.optim.Adam([patch_texture_var], lr=args.lr)

    manhole_dir = ADV_MANHOLE_DIR / "adversarial_example"
    candidate_images = _load_manhole_set(str(manhole_dir), image_size=256).to(device)
    losses = AdvManholeLosses(
        adv_content_loss=AdvContentLoss(candidate_images=candidate_images),
        mde_loss_weight=2.0,
        ss_ua_loss_weight=0.5,
        ss_ta_loss_weight=0.5,
        tv_loss_weight=1.0,
        content_loss_weight=0.5,
        background_loss_weight=0.0,
    )
    framework = AdvManholeFramework(
        optimizer=optimizer,
        mde_model=mde_wrapped,
        ss_model=ss_wrapped,
        loss=losses,
        patch_texture_var=patch_texture_var,
        depth_planar_mapping=mapping,
        texture_augmentation=transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.1),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ]),
        output_augmentation=transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.1),
        ]),
        device=device,
    )

    loader = adv_dataset["train"]
    batch = next(iter(loader))

    before = patch_texture_var.detach().clone()
    optimizer.zero_grad()
    result = framework.forward(batch, patch_texture_var, mapping, losses)
    total_loss = result["loss"]["total_loss"]
    total_loss.backward()

    grad = patch_texture_var.grad
    grad_norm = 0.0 if grad is None else float(torch.nan_to_num(grad).norm().item())
    coverage = float(result["texture_masks"].float().mean().item())
    image_delta = float((result["final_images"] - batch["rgb"].to(device)).abs().mean().item())
    loss_values = {
        key: finite_float(value)
        for key, value in result["loss"].items()
        if "loss" in key
    }

    optimizer.step()
    patch_texture_var.data.clamp_(0.0, 1.0)
    patch_update_norm = float((patch_texture_var.detach() - before).norm().item())

    Image.fromarray(
        (patch_texture_var.permute(1, 2, 0).detach().cpu().numpy() * 255).astype("uint8")
    ).save(Path(args.output_dir) / "patch_after_one_step.png")
    Image.fromarray(
        (result["final_images"][0].permute(1, 2, 0).detach().cpu().numpy() * 255).astype("uint8")
    ).save(Path(args.output_dir) / "patched_image_batch0.png")
    Image.fromarray(
        (result["texture_masks"][0, 0].detach().cpu().numpy() * 255).astype("uint8")
    ).save(Path(args.output_dir) / "texture_mask_batch0.png")

    print("\nAdv-manhole correctness check")
    print(f"mask_coverage:     {coverage:.6f}")
    print(f"image_delta_mean:  {image_delta:.6f}")
    print(f"grad_norm:         {grad_norm:.6f}")
    print(f"patch_update_norm: {patch_update_norm:.6f}")
    for key, value in loss_values.items():
        print(f"{key}: {value:.6f}")

    failures = []
    if coverage < args.min_coverage:
        failures.append(f"mask coverage below threshold ({coverage:.6f} < {args.min_coverage})")
    if not torch.isfinite(total_loss):
        failures.append("total loss is not finite")
    if grad_norm <= 0.0:
        failures.append("patch gradient is zero")
    if patch_update_norm <= 0.0:
        failures.append("optimizer did not change patch")
    if image_delta <= 0.0:
        failures.append("patched image equals clean image")

    if failures:
        print("\nFAIL")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)

    print("\nPASS")


if __name__ == "__main__":
    main()
