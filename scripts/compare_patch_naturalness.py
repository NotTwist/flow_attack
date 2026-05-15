#!/usr/bin/env python3
"""Compare simple naturalness/stealth metrics for saved patches.

Metrics:
  - TV: lower means smoother.
  - NPS: lower means closer to a coarse printable RGB palette.
  - manhole_vgg_content: lower means closer to adv-manhole reference manholes.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "attacks" / "adversarial_manhole"))

from adv_manhole.attack.naturalness import AdvContentLoss  # noqa: E402
from run_adv_manhole_kitti import _load_manhole_set  # noqa: E402


def load_patch(path: Path, size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    return transforms.ToTensor()(img)


def tv_metric(patch: torch.Tensor) -> float:
    x = patch.unsqueeze(0)
    dx = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    dy = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    return float((dx + dy).item())


def nps_metric(patch: torch.Tensor, grid_size: int = 6) -> float:
    vals = torch.linspace(0.05, 0.95, steps=grid_size, device=patch.device)
    printable = torch.stack(torch.meshgrid(vals, vals, vals, indexing="ij"), dim=-1).reshape(-1, 3)
    pixels = patch.permute(1, 2, 0).reshape(-1, 3)
    dist = torch.sqrt(((pixels.unsqueeze(1) - printable.unsqueeze(0)) ** 2).sum(dim=2) + 1e-8)
    return float(dist.min(dim=1).values.mean().item())


def discover_default_patches() -> list[tuple[str, Path]]:
    candidates = []
    known = [
        ("adv_manhole_smoke", Path("experiment_data/adv_manhole_kitti_smoke/trained_patch_kitti.png")),
        ("adv_manhole_one_step", Path("experiment_data/adv_manhole_kitti_correctness_check/patch_after_one_step.png")),
        ("diffusion_trained", Path("experiment_data/diffusion_patch_kitti15_no_projection_20260428_175503/patch_checkpoints/patch_epoch_010.png")),
        ("pixel_trained", Path("experiment_data/pixel_patch_kitti15_no_projection_20260429_201754/patch_checkpoints/patch_epoch_010.png")),
        ("diffusion_base_dog", Path("test_assets/dog.jpg")),
    ]
    for label, path in known:
        if path.exists():
            candidates.append((label, path))
    return candidates


def save_contact_sheet(rows: list[dict], output_path: Path, size: int) -> None:
    if not rows:
        return
    cell_w = size
    label_h = 34
    sheet = Image.new("RGB", (cell_w * len(rows), size + label_h), "white")
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(rows):
        img = Image.open(row["path"]).convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
        x = i * cell_w
        sheet.paste(img, (x, 0))
        draw.text((x + 4, size + 4), row["label"][:28], fill=(0, 0, 0))
    sheet.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=Path("experiment_data/patch_naturalness_comparison"))
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--patch", nargs=2, action="append", metavar=("LABEL", "PATH"), default=[])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    refs = _load_manhole_set(
        str(REPO_ROOT / "attacks" / "adversarial_manhole" / "adversarial_example"),
        image_size=args.size,
    ).to(device)
    content_loss = AdvContentLoss(candidate_images=refs, device=device)

    patch_specs = [(label, Path(path)) for label, path in args.patch] or discover_default_patches()
    rows = []
    for label, path in patch_specs:
        if not path.exists():
            print(f"Skipping missing patch: {label} -> {path}")
            continue
        patch = load_patch(path, args.size).to(device)
        patch_batch = patch.unsqueeze(0).requires_grad_(True)
        manhole_loss, nearest_idx = content_loss(patch_batch)
        row = {
            "label": label,
            "path": str(path),
            "tv": tv_metric(patch.detach().cpu()),
            "nps": nps_metric(patch.detach().cpu()),
            "manhole_vgg_content": float(manhole_loss.item()),
            "nearest_manhole_ref_idx": int(nearest_idx.item()),
        }
        rows.append(row)

    csv_path = args.output_dir / "patch_naturalness_metrics.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["label", "path", "tv", "nps", "manhole_vgg_content", "nearest_manhole_ref_idx"],
        )
        writer.writeheader()
        writer.writerows(rows)

    save_contact_sheet(rows, args.output_dir / "patch_naturalness_contact_sheet.png", args.size)

    print(f"Saved metrics: {csv_path}")
    for row in rows:
        print(
            f"{row['label']}: tv={row['tv']:.4f}, nps={row['nps']:.4f}, "
            f"manhole_content={row['manhole_vgg_content']:.4f}"
        )


if __name__ == "__main__":
    main()
