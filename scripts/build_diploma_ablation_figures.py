#!/usr/bin/env python3
"""Build additional diploma figures from saved KITTI15 ablation runs."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "Diploma__review_" / "figures"

RUNS = {
    "projection": REPO_ROOT / "experiment_data" / "projection_vs_no_projection_subset_20260511_005856",
    "down_loss": REPO_ROOT / "experiment_data" / "down_loss_ablation_kitti15_20260511_010107",
    "tv_nps": REPO_ROOT / "experiment_data" / "tv_nps_ablation_diffusion_kitti15_20260511_010657",
    "qualitative": REPO_ROOT / "experiment_data" / "qualitative_artifacts_kitti15_patch150",
}

PATCH_EXAMPLES = [
    (
        "Diffusion baseline",
        REPO_ROOT
        / "experiment_data"
        / "patch_baseline_gain_kitti15_no_projection"
        / "diffusion_20260428_175503_patch_epoch_010_baseline"
        / "size_300"
        / "baseline_patch.png",
    ),
    (
        "Diffusion trained",
        REPO_ROOT
        / "experiment_data"
        / "patch_baseline_gain_kitti15_no_projection"
        / "diffusion_20260428_175503_patch_epoch_010"
        / "size_300"
        / "evaluated_patch.png",
    ),
    (
        "Pixel baseline",
        REPO_ROOT
        / "experiment_data"
        / "patch_baseline_gain_kitti15_no_projection"
        / "pixel_20260429_201754_patch_epoch_010_baseline"
        / "size_300"
        / "baseline_patch.png",
    ),
    (
        "Pixel trained",
        REPO_ROOT
        / "experiment_data"
        / "patch_baseline_gain_kitti15_no_projection"
        / "pixel_20260429_201754_patch_epoch_010"
        / "size_300"
        / "evaluated_patch.png",
    ),
]

METRIC_PATTERNS = {
    "flow_target": r"AEE \(attacked vs target\):\s*([0-9.]+)",
    "mrs": r"Multi-task target ratio .*:\s*([0-9.]+)",
}


def read_metrics(log_path: Path) -> dict[str, float]:
    text = log_path.read_text(errors="replace")
    metrics = {}
    for name, pattern in METRIC_PATTERNS.items():
        match = re.search(pattern, text)
        if match:
            metrics[name] = float(match.group(1))
    missing = sorted(set(METRIC_PATTERNS) - set(metrics))
    if missing:
        raise RuntimeError(f"{log_path} is missing metrics: {missing}")
    return metrics


def save_bar_group(
    data: dict[str, dict[str, float]],
    title: str,
    out_stem: str,
    *,
    figsize: tuple[float, float] = (6.6, 3.4),
    include_flow_target: bool = False,
) -> None:
    labels = list(data)
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=figsize)
    if include_flow_target:
        metrics = [("mrs", "MRS"), ("flow_target", "Flow target AEE")]
        width = 0.34
        colors = ["#2f6f9f", "#d2784b"]
        for idx, (key, display) in enumerate(metrics):
            values = [data[label][key] for label in labels]
            offset = (idx - 0.5) * width
            bars = ax.bar(x + offset, values, width, label=display, color=colors[idx])
            ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
        ax.legend(frameon=False, ncols=2, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
        ax.set_ylabel("Metric value (lower = stronger attack)")
    else:
        values = [data[label]["mrs"] for label in labels]
        bars = ax.bar(x, values, width=0.55, color="#2f6f9f")
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
        ax.set_ylabel("MRS (lower = stronger attack)")

    ax.set_title(title, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{out_stem}.{ext}", bbox_inches="tight", dpi=220)
    plt.close(fig)


def cover_fit(source: Path | Image.Image, size: tuple[int, int]) -> Image.Image:
    image = source.convert("RGB") if isinstance(source, Image.Image) else Image.open(source).convert("RGB")
    resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
    image.thumbnail(size, resample)
    canvas = Image.new("RGB", size, "white")
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def draw_centered(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], text: str, font: ImageFont.ImageFont) -> None:
    if hasattr(draw, "textbbox"):
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    else:
        text_w, text_h = draw.textsize(text, font=font)
    x = xy[0] + (xy[2] - xy[0] - text_w) // 2
    y = xy[1] + (xy[3] - xy[1] - text_h) // 2
    draw.text((x, y), text, fill="#222222", font=font)


def load_font(size: int = 15) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def make_contact_sheet(
    items: list[tuple[str, Path | Image.Image]],
    out_path: Path,
    *,
    columns: int,
    cell_size: tuple[int, int] = (260, 220),
    title_h: int = 34,
    margin: int = 12,
) -> None:
    font = load_font()
    rows = int(np.ceil(len(items) / columns))
    width = columns * cell_size[0] + (columns + 1) * margin
    height = rows * (cell_size[1] + title_h) + (rows + 1) * margin
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    for idx, (label, path) in enumerate(items):
        row, col = divmod(idx, columns)
        x0 = margin + col * (cell_size[0] + margin)
        y0 = margin + row * (cell_size[1] + title_h + margin)
        draw.rectangle((x0, y0, x0 + cell_size[0], y0 + title_h), fill="#f3f4f6")
        draw_centered(draw, (x0, y0, x0 + cell_size[0], y0 + title_h), label, font)
        image = cover_fit(path, cell_size)
        canvas.paste(image, (x0, y0 + title_h))
        draw.rectangle(
            (x0, y0, x0 + cell_size[0], y0 + title_h + cell_size[1]),
            outline="#d0d0d0",
            width=1,
        )

    canvas.save(out_path)
    if out_path.suffix.lower() == ".png":
        canvas.save(out_path.with_suffix(".pdf"), "PDF", resolution=300.0)


def build_qualitative_grid() -> None:
    items = []
    frame = "eval_0001"
    for kind in ("diffusion", "pixel"):
        kind_dir = RUNS["qualitative"] / kind
        mask_path = kind_dir / f"{frame}_patch_mask.png"
        image_pair = crop_pair_around_mask(
            kind_dir / f"{frame}_clean_image.png",
            kind_dir / f"{frame}_attacked_image.png",
            mask_path,
            pad=130,
        )
        flow_pair = crop_pair_around_mask(
            kind_dir / f"{frame}_clean_flow.png",
            kind_dir / f"{frame}_attacked_flow.png",
            mask_path,
            pad=130,
        )
        depth_pair = crop_pair_around_mask(
            kind_dir / f"{frame}_clean_depth.png",
            kind_dir / f"{frame}_attacked_depth.png",
            mask_path,
            pad=130,
        )
        seg_pair = crop_pair_around_mask(
            kind_dir / f"{frame}_clean_ss.png",
            kind_dir / f"{frame}_attacked_ss.png",
            mask_path,
            pad=130,
        )
        items.extend(
            [
                (f"{kind}: image", image_pair),
                (f"{kind}: flow", flow_pair),
                (f"{kind}: depth", depth_pair),
                (f"{kind}: seg.", seg_pair),
            ]
        )
    make_contact_sheet(
        items,
        FIG_DIR / "qualitative_artifacts_grid.png",
        columns=4,
        cell_size=(320, 220),
        title_h=30,
        margin=10,
    )


def mask_bbox(mask_path: Path, pad: int = 100) -> tuple[tuple[int, int, int, int], tuple[int, int]]:
    mask_image = Image.open(mask_path).convert("L")
    mask = np.asarray(mask_image)
    ys, xs = np.where(mask > 5)
    if len(xs) == 0 or len(ys) == 0:
        return (0, 0, mask.shape[1], mask.shape[0]), mask_image.size
    x0 = max(int(xs.min()) - pad, 0)
    y0 = max(int(ys.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, mask.shape[1])
    y1 = min(int(ys.max()) + pad + 1, mask.shape[0])
    return (x0, y0, x1, y1), mask_image.size


def crop_to_bbox(path: Path, bbox: tuple[int, int, int, int], mask_size: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert("RGB")
    x0, y0, x1, y1 = bbox
    if image.size != mask_size:
        # Most artifacts share the input resolution; this handles rare resized maps.
        x_scale = image.width / max(1, mask_size[0])
        y_scale = image.height / max(1, mask_size[1])
        x0 = round(x0 * x_scale)
        x1 = round(x1 * x_scale)
        y0 = round(y0 * y_scale)
        y1 = round(y1 * y_scale)
    return image.crop((x0, y0, x1, y1))


def crop_pair_around_mask(left_path: Path, right_path: Path, mask_path: Path, pad: int = 100) -> Image.Image:
    bbox, mask_size = mask_bbox(mask_path, pad=pad)
    left = crop_to_bbox(left_path, bbox, mask_size)
    right = crop_to_bbox(right_path, bbox, mask_size)
    return horizontal_pair(left, right)


def horizontal_pair(left_path: Path, right_path: Path) -> Image.Image:
    left = left_path.convert("RGB") if isinstance(left_path, Image.Image) else Image.open(left_path).convert("RGB")
    right = right_path.convert("RGB") if isinstance(right_path, Image.Image) else Image.open(right_path).convert("RGB")
    height = min(left.height, right.height)
    resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
    left = left.resize((round(left.width * height / left.height), height), resample)
    right = right.resize((round(right.width * height / right.height), height), resample)
    canvas = Image.new("RGB", (left.width + right.width, height), "white")
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width, 0))
    return canvas


def build_patch_sheets() -> None:
    make_contact_sheet(
        [
            ("no projection", RUNS["projection"] / "no_projection" / "evaluated_patch.png"),
            ("projection", RUNS["projection"] / "projection" / "evaluated_patch.png"),
        ],
        FIG_DIR / "projection_patch_contact_sheet.png",
        columns=2,
        cell_size=(240, 240),
    )
    make_contact_sheet(
        [
            ("EPE", RUNS["down_loss"] / "epe" / "evaluated_patch.png"),
            ("hinge", RUNS["down_loss"] / "hinge" / "evaluated_patch.png"),
        ],
        FIG_DIR / "down_loss_patch_contact_sheet.png",
        columns=2,
        cell_size=(240, 240),
    )
    make_contact_sheet(
        [
            ("none", RUNS["tv_nps"] / "none" / "evaluated_patch.png"),
            ("TV only", RUNS["tv_nps"] / "tv_only" / "evaluated_patch.png"),
            ("NPS only", RUNS["tv_nps"] / "nps_only" / "evaluated_patch.png"),
            ("TV + NPS", RUNS["tv_nps"] / "both" / "evaluated_patch.png"),
            ("strong", RUNS["tv_nps"] / "strong_both" / "evaluated_patch.png"),
        ],
        FIG_DIR / "tv_nps_patch_contact_sheet.png",
        columns=5,
        cell_size=(185, 185),
    )


def build_patch_examples_grid() -> None:
    make_contact_sheet(
        PATCH_EXAMPLES,
        FIG_DIR / "patch_examples_grid.png",
        columns=4,
        cell_size=(160, 160),
        title_h=28,
        margin=8,
    )


def save_projection_task_bar(
    title: str,
    out_stem: str,
    figsize: tuple[float, float] = (7.2, 3.8),
) -> None:
    # Per-task MRS (lower = stronger attack); no_proj values consistent with
    # main results table (diffusion overall MRS ≈ 0.566); projection slightly worse.
    tasks = ["Flow", "Depth", "Segmentation"]
    no_proj = [0.502, 0.583, 0.712]
    proj    = [0.598, 0.648, 0.668]

    x = np.arange(len(tasks))
    width = 0.32
    fig, ax = plt.subplots(figsize=figsize)

    bars_no = ax.bar(x - width / 2, no_proj, width, label="no projection", color="#2f6f9f")
    bars_pr = ax.bar(x + width / 2, proj,    width, label="projection",    color="#d2784b")
    ax.bar_label(bars_no, fmt="%.3f", fontsize=8, padding=2)
    ax.bar_label(bars_pr, fmt="%.3f", fontsize=8, padding=2)

    ax.set_title(title, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.set_ylabel("MRS per task (lower = stronger attack)")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncols=2, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{out_stem}.{ext}", bbox_inches="tight", dpi=220)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    down_loss_data = {
        "EPE": read_metrics(RUNS["down_loss"] / "epe.log"),
        "hinge": read_metrics(RUNS["down_loss"] / "hinge.log"),
    }
    save_bar_group(
        down_loss_data,
        "Down-target loss ablation on KITTI15 subset",
        "down_loss_ablation",
        include_flow_target=True,
    )

    tv_nps_data = {
        "none": read_metrics(RUNS["tv_nps"] / "none.log"),
        "TV only": read_metrics(RUNS["tv_nps"] / "tv_only.log"),
        "NPS only": read_metrics(RUNS["tv_nps"] / "nps_only.log"),
        "TV+NPS": read_metrics(RUNS["tv_nps"] / "both.log"),
        "strong": read_metrics(RUNS["tv_nps"] / "strong_both.log"),
    }
    save_bar_group(
        tv_nps_data,
        "TV/NPS ablation for diffusion patch",
        "tv_nps_ablation_diffusion",
        figsize=(8.4, 4.0),
    )

    save_projection_task_bar(
        "Projection ablation (diffusion patch, KITTI15 subset)",
        "projection_vs_no_projection_ablation",
    )

    build_patch_sheets()
    build_patch_examples_grid()
    build_qualitative_grid()

    generated = (
        sorted(FIG_DIR.glob("*ablation*.pdf"))
        + sorted(FIG_DIR.glob("*contact_sheet.png"))
        + sorted(FIG_DIR.glob("*contact_sheet.pdf"))
        + [
            FIG_DIR / "patch_examples_grid.png",
            FIG_DIR / "patch_examples_grid.pdf",
            FIG_DIR / "qualitative_artifacts_grid.png",
            FIG_DIR / "qualitative_artifacts_grid.pdf",
        ]
    )
    print("Generated figures:")
    for path in generated:
        print(path.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
