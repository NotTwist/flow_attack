#!/usr/bin/env python3
"""Measure attacked three-model inference FPS across image resolutions."""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", f"/tmp/matplotlib-{os.environ.get('USER', 'codex')}")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
import ptlflow.utils.io_adapter

from attacks.DetectionDefenses.helper_functions.patch_adversary import PatchAdversary, circ_mask
from datasets_utils.dataset_utils import prepare_dataloader
from models.model_utils import load_mde_model, load_seg_model
from run_patch_attack import load_model
from utils.process_images import replace_images_dic


def parse_resolutions(spec: str) -> list[tuple[int, int]]:
    out = []
    for item in spec.replace(";", ",").split(","):
        item = item.strip().lower()
        if not item:
            continue
        if "x" not in item:
            raise argparse.ArgumentTypeError(
                f"Resolution '{item}' must be HEIGHTxWIDTH, e.g. 384x1248"
            )
        h_s, w_s = item.split("x", 1)
        h, w = int(h_s), int(w_s)
        if h <= 0 or w <= 0:
            raise argparse.ArgumentTypeError(f"Resolution must be positive: {item}")
        out.append((h, w))
    if not out:
        raise argparse.ArgumentTypeError("At least one resolution is required")
    return out


def make_patch(args, image_size: tuple[int, int], device: torch.device) -> PatchAdversary:
    patch_size = min(args.patch_size, image_size[0] - 1, image_size[1] - 1)
    if patch_size < 2:
        raise ValueError(f"Resolution {image_size} is too small for a patch benchmark")

    if args.trained_patch:
        patch = PatchAdversary(
            args.trained_patch,
            size=patch_size,
            angle=0,
            scale=1,
            random_location=not args.fixed_patch_location,
            image_size=image_size,
            ellipse_scale_y=args.y_scale,
        ).to(device)
    else:
        rgb = torch.rand(1, 3, patch_size, patch_size)
        mask = circ_mask(rgb)
        patch_tensor = torch.cat([rgb, mask], dim=1)
        patch = PatchAdversary(
            patch_tensor,
            size=patch_size,
            angle=0,
            scale=1,
            random_location=not args.fixed_patch_location,
            image_size=image_size,
            ellipse_scale_y=args.y_scale,
        ).to(device)
    patch.eval()
    for param in patch.parameters():
        param.requires_grad_(False)
    return patch


def resize_image_pair(images: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Resize [B,2,C,H,W] image pairs on CPU/GPU preserving pair layout."""
    b, t, c, _, _ = images.shape
    flat = images.reshape(b * t, c, images.shape[-2], images.shape[-1]).float()
    resized = F.interpolate(flat, size=size, mode="bilinear", align_corners=False)
    return resized.reshape(b, t, c, size[0], size[1]).clamp(0.0, 1.0)


def build_samples(
    *,
    args,
    flow_model,
    loader,
    resolution: tuple[int, int],
    device: torch.device,
) -> list[dict[str, torch.Tensor | dict]]:
    samples = []
    io_adapter = ptlflow.utils.io_adapter.IOAdapter(
        flow_model,
        input_size=resolution,
        cuda=device.type == "cuda",
    )
    for batch_idx, (images, _flow, _valid, _meta, _K) in enumerate(loader):
        if batch_idx >= args.num_batches:
            break
        images = resize_image_pair(images, resolution)
        inputs = io_adapter.prepare_inputs(inputs={"images": images})
        samples.append({"images": inputs["images"], "inputs_template": inputs})
    return samples


@torch.no_grad()
def run_attacked_three_model_inference(
    sample: dict,
    *,
    patch: PatchAdversary,
    flow_model,
    mde_model,
    ss_model,
    flow_shift: float,
):
    images = sample["images"]
    inputs_template = sample["inputs_template"]

    image1 = images[:, 0]
    image2 = images[:, 1]
    attacked_image1, attacked_image2, _mask, _y, _x = patch(
        image1,
        image2,
        flow_shift=flow_shift,
    )
    attacked_pair = torch.stack([attacked_image1, attacked_image2], dim=1)[0]
    inputs = replace_images_dic(inputs_template, attacked_pair, clone=True)

    flow_pred = flow_model(inputs)["flows"]
    depth_pred = mde_model(inputs)
    seg_pred = ss_model(inputs)

    # Touch outputs so lazy execution cannot disappear behind the timer.
    return flow_pred, depth_pred, seg_pred


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_once(device: torch.device, fn) -> float:
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize(device)
        return float(start.elapsed_time(end)) / 1000.0

    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def benchmark_resolution(args, models, loader, resolution, device):
    flow_model, mde_model, ss_model = models
    h, w = resolution
    samples = build_samples(
        args=args,
        flow_model=flow_model,
        loader=loader,
        resolution=resolution,
        device=device,
    )
    if not samples:
        raise RuntimeError("No samples were loaded for benchmarking")

    patch = make_patch(args, image_size=resolution, device=device)

    def run_all_samples():
        for sample in samples:
            run_attacked_three_model_inference(
                sample,
                patch=patch,
                flow_model=flow_model,
                mde_model=mde_model,
                ss_model=ss_model,
                flow_shift=args.flow_shift,
            )

    for _ in range(args.warmup):
        run_all_samples()
    synchronize(device)

    timings = [time_once(device, run_all_samples) for _ in range(args.repeats)]
    images_per_repeat = len(samples)
    per_image = [t / images_per_repeat for t in timings]
    mean_latency = statistics.mean(per_image)
    std_latency = statistics.pstdev(per_image) if len(per_image) > 1 else 0.0
    fps = 1.0 / mean_latency if mean_latency > 0 else math.nan

    return {
        "height": h,
        "width": w,
        "megapixels": h * w / 1_000_000.0,
        "num_batches": len(samples),
        "repeats": args.repeats,
        "latency_ms": mean_latency * 1000.0,
        "latency_std_ms": std_latency * 1000.0,
        "fps": fps,
        "status": "ok",
    }


def write_csv(rows: list[dict], output_dir: Path) -> Path:
    path = output_dir / "attack_fps_resolution_sweep.csv"
    fieldnames = [
        "height",
        "width",
        "megapixels",
        "num_batches",
        "repeats",
        "latency_ms",
        "latency_std_ms",
        "fps",
        "status",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def plot(rows: list[dict], output_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    import numpy as np

    ok_rows = [r for r in rows if r.get("status") == "ok" and math.isfinite(float(r["fps"]))]
    if not ok_rows:
        return []

    x = np.array([float(r["megapixels"]) for r in ok_rows])
    fps = np.array([float(r["fps"]) for r in ok_rows])
    lat = np.array([float(r["latency_ms"]) for r in ok_rows])
    labels = [f"{int(r['height'])}x{int(r['width'])}" for r in ok_rows]

    saved = []
    fps_path = output_dir / "attack_fps_vs_resolution.png"
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, fps, marker="o")
    for xi, yi, label in zip(x, fps, labels):
        ax.annotate(label, (xi, yi), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel("Resolution, megapixels")
    ax.set_ylabel("FPS, attacked 3-model inference")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fps_path, dpi=180)
    plt.close(fig)
    saved.append(fps_path)

    latency_path = output_dir / "attack_latency_vs_resolution.png"
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, lat, marker="o", color="tab:orange")
    for xi, yi, label in zip(x, lat, labels):
        ax.annotate(label, (xi, yi), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel("Resolution, megapixels")
    ax.set_ylabel("Latency per image pair, ms")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(latency_path, dpi=180)
    plt.close(fig)
    saved.append(latency_path)

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="Kitti15", choices=["Kitti15", "Sintel", "carla"])
    parser.add_argument("--eval_mode", default="testing", choices=["training", "testing"])
    parser.add_argument("--model_name", default="raft")
    parser.add_argument("--mde_model", default="depth-anything-v2", choices=["depth-anything-v2", "marigold"])
    parser.add_argument("--ss_model", default="segformer_cityscapes")
    parser.add_argument(
        "--resolutions",
        type=parse_resolutions,
        default=parse_resolutions("192x640,256x832,320x1024,375x1242"),
        help="Comma-separated HEIGHTxWIDTH list.",
    )
    parser.add_argument("--num_batches", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--patch_size", type=int, default=100)
    parser.add_argument("--trained_patch", default="", help="Optional RGBA patch PNG.")
    parser.add_argument("--fixed_patch_location", action="store_true")
    parser.add_argument("--flow_shift", type=float, default=0.0)
    parser.add_argument("--y_scale", type=float, default=3.0)
    parser.add_argument("--output_dir", default="experiment_data/attack_fps_resolution_sweep")
    args = parser.parse_args()

    if args.num_batches <= 0:
        raise SystemExit("--num_batches must be positive")
    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Using device: {device}")

    loader, _ = prepare_dataloader(
        mode=args.eval_mode,
        dataset_name=args.dataset,
        subset_size=args.num_batches,
        n_images=1,
        has_depth=False,
    )

    print("Loading models...")
    flow_model = load_model(args.model_name, args.dataset.lower()).to(device).eval()
    mde_model = load_mde_model(model_name=args.mde_model, device=device).eval()
    ss_model = load_seg_model(model_name=args.ss_model, device=device).eval()
    for model in (flow_model, mde_model, ss_model):
        for param in model.parameters():
            param.requires_grad_(False)

    rows = []
    for resolution in args.resolutions:
        h, w = resolution
        print(f"Benchmarking {h}x{w}...")
        try:
            row = benchmark_resolution(
                args,
                (flow_model, mde_model, ss_model),
                loader,
                resolution,
                device,
            )
            print(
                f"  FPS={row['fps']:.3f}, latency={row['latency_ms']:.2f} ms "
                f"over {row['num_batches']} image pairs"
            )
        except RuntimeError as exc:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            row = {
                "height": h,
                "width": w,
                "megapixels": h * w / 1_000_000.0,
                "num_batches": args.num_batches,
                "repeats": args.repeats,
                "latency_ms": math.nan,
                "latency_std_ms": math.nan,
                "fps": math.nan,
                "status": f"failed: {exc}",
            }
            print(f"  Failed: {exc}")
        rows.append(row)

    csv_path = write_csv(rows, output_dir)
    plot_paths = plot(rows, output_dir)

    print(f"Wrote {csv_path}")
    for path in plot_paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
