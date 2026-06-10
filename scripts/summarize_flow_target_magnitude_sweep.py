#!/usr/bin/env python3
"""Summarize a patch flow-target magnitude sweep from run logs."""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
from pathlib import Path


os.environ.setdefault("MPLCONFIGDIR", f"/tmp/matplotlib-{os.environ.get('USER', 'codex')}")


RE_MAG = re.compile(r"mag_([-+]?\d+(?:p\d+)?)")
RE_ANGLE = re.compile(r"angle_([-+]?\d+(?:p\d+)?)")
RE_AEE_TARGET = re.compile(r"AEE \(attacked vs target\):\s*(?P<v>[-+.\deE]+|N/A)")
RE_AEE_ATTACK = re.compile(r"AEE \(init vs attack\):\s*(?P<v>[-+.\deE]+|N/A)")
RE_MRS = re.compile(r"Multi-task target ratio .*:\s*(?P<v>[-+.\deE]+|N/A)")
RE_MDE_ATTACK = re.compile(r"MDE RMSE \(attacked vs target\):\s*(?P<v>[-+.\deE]+|N/A)")
RE_IOU_TARGET = re.compile(r"IoU \(attacked vs target\):\s*(?P<v>[-+.\deE]+|N/A)")


def parse_float(value: str) -> float:
    if value == "N/A":
        return math.nan
    return float(value)


def last_match(pattern: re.Pattern[str], text: str) -> float:
    matches = list(pattern.finditer(text))
    if not matches:
        return math.nan
    return parse_float(matches[-1].group("v"))


def magnitude_from_log(path: Path, text: str) -> float:
    cli_match = re.search(r"--flow_target_magnitude\s+([-+]?\d+(?:\.\d+)?)", text)
    if cli_match:
        return float(cli_match.group(1))

    name_match = RE_MAG.search(path.stem)
    if name_match:
        return float(name_match.group(1).replace("p", "."))

    raise ValueError(f"Could not infer magnitude for {path}")


def angle_from_log(text: str) -> float:
    cli_match = re.search(r"--flow_target_angle_deg\s+([-+]?\d+(?:\.\d+)?)", text)
    if cli_match:
        return float(cli_match.group(1))
    return math.nan


def angle_from_path(path: Path) -> float:
    name_match = RE_ANGLE.search(path.stem)
    if name_match:
        return float(name_match.group(1).replace("p", "."))
    return math.nan


def collect(root: Path) -> list[dict[str, float | str]]:
    rows = []
    for log_path in sorted(root.glob("*.log")):
        if "mag_" not in log_path.stem:
            continue
        text = log_path.read_text(encoding="utf-8", errors="replace")
        magnitude = magnitude_from_log(log_path, text)
        angle = angle_from_log(text)
        if not math.isfinite(angle):
            angle = angle_from_path(log_path)
        rows.append(
            {
                "flow_target_magnitude": magnitude,
                "flow_target_angle_deg": angle,
                "aee_target_attack": last_match(RE_AEE_TARGET, text),
                "aee_init_attack": last_match(RE_AEE_ATTACK, text),
                "multitask_robustness_score": last_match(RE_MRS, text),
                "mde_rmse_target_attack": last_match(RE_MDE_ATTACK, text),
                "iou_target_attack": last_match(RE_IOU_TARGET, text),
                "log_path": str(log_path),
            }
        )
    rows.sort(key=lambda r: (float(r["flow_target_angle_deg"]), float(r["flow_target_magnitude"])))
    return rows


def write_csv(root: Path, rows: list[dict[str, float | str]]) -> Path:
    csv_path = root / "flow_target_magnitude_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "flow_target_magnitude",
                "flow_target_angle_deg",
                "aee_target_attack",
                "aee_init_attack",
                "multitask_robustness_score",
                "mde_rmse_target_attack",
                "iou_target_attack",
                "log_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def save_figure(fig, png_path: Path) -> list[Path]:
    pdf_path = png_path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=180)
    fig.savefig(pdf_path)
    return [png_path, pdf_path]


def plot_metric(root: Path, x, y, *, name: str, ylabel: str, title: str, log_x: bool = True):
    import matplotlib.pyplot as plt
    import numpy as np

    finite = np.isfinite(x) & np.isfinite(y)
    if not finite.any():
        return []

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x[finite], y[finite], marker="o")
    ax.set_xlabel("Target flow magnitude, px")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if log_x and np.all(x[finite] > 0):
        ax.set_xscale("log")
    fig.tight_layout()
    paths = save_figure(fig, root / f"{name}.png")
    plt.close(fig)
    return paths


def plot_metric_by_angle(root: Path, rows, metric_key: str, *, name: str, ylabel: str, title: str):
    import matplotlib.pyplot as plt
    import numpy as np

    angles = sorted({float(r["flow_target_angle_deg"]) for r in rows if math.isfinite(float(r["flow_target_angle_deg"]))})
    if not angles:
        return []

    fig, ax = plt.subplots(figsize=(8, 4.8))
    plotted = False
    for angle in angles:
        angle_rows = [
            r for r in rows
            if math.isclose(float(r["flow_target_angle_deg"]), angle)
            and math.isfinite(float(r[metric_key]))
        ]
        if not angle_rows:
            continue
        angle_rows.sort(key=lambda r: float(r["flow_target_magnitude"]))
        x = np.array([float(r["flow_target_magnitude"]) for r in angle_rows], dtype=float)
        y = np.array([float(r[metric_key]) for r in angle_rows], dtype=float)
        ax.plot(x, y, marker="o", label=f"{angle:g} deg")
        plotted = True

    if not plotted:
        plt.close(fig)
        return []

    ax.set_xlabel("Target flow magnitude, px")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if all(float(r["flow_target_magnitude"]) > 0 for r in rows):
        ax.set_xscale("log")
    ax.legend(title="Direction", fontsize=8)
    fig.tight_layout()
    paths = save_figure(fig, root / f"{name}.png")
    plt.close(fig)
    return paths


def plot_heatmap(root: Path, rows, metric_key: str, *, name: str, label: str, title: str):
    import matplotlib.pyplot as plt
    import numpy as np

    angles = sorted({float(r["flow_target_angle_deg"]) for r in rows if math.isfinite(float(r["flow_target_angle_deg"]))})
    magnitudes = sorted({float(r["flow_target_magnitude"]) for r in rows})
    if not angles or not magnitudes:
        return []

    grid = np.full((len(angles), len(magnitudes)), np.nan, dtype=float)
    angle_to_i = {v: i for i, v in enumerate(angles)}
    mag_to_j = {v: j for j, v in enumerate(magnitudes)}
    for r in rows:
        angle = float(r["flow_target_angle_deg"])
        mag = float(r["flow_target_magnitude"])
        value = float(r[metric_key])
        if math.isfinite(angle) and math.isfinite(value):
            grid[angle_to_i[angle], mag_to_j[mag]] = value

    if not np.isfinite(grid).any():
        return []

    fig, ax = plt.subplots(figsize=(9, 5.2))
    masked = np.ma.masked_invalid(grid)
    im = ax.imshow(masked, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(np.arange(len(magnitudes)))
    ax.set_xticklabels([f"{v:g}" for v in magnitudes], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(angles)))
    ax.set_yticklabels([f"{v:g}" for v in angles])
    ax.set_xlabel("Target flow magnitude, px")
    ax.set_ylabel("Target angle, degrees")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(label)
    fig.tight_layout()
    paths = save_figure(fig, root / f"{name}.png")
    plt.close(fig)
    return paths


def maybe_plot(root: Path, rows: list[dict[str, float | str]]) -> list[Path]:
    try:
        import numpy as np
    except Exception as exc:
        print(f"Skipping plots: numpy unavailable ({exc})")
        return []

    x = np.array([float(r["flow_target_magnitude"]) for r in rows], dtype=float)
    target_aee = np.array([float(r["aee_target_attack"]) for r in rows], dtype=float)
    delta_aee = np.array([float(r["aee_init_attack"]) for r in rows], dtype=float)
    mrs = np.array([float(r["multitask_robustness_score"]) for r in rows], dtype=float)
    mde = np.array([float(r["mde_rmse_target_attack"]) for r in rows], dtype=float)
    iou = np.array([float(r["iou_target_attack"]) for r in rows], dtype=float)

    saved = []
    saved.extend(plot_metric(
        root,
        x,
        target_aee,
        name="flow_target_magnitude_target_aee",
        ylabel="Flow target AEE (lower is better)",
        title="Targeted flow success vs target magnitude",
    ))
    saved.extend(plot_metric(
        root,
        x,
        delta_aee,
        name="flow_target_magnitude_delta_aee",
        ylabel="AEE clean vs attacked (higher is stronger change)",
        title="Change from clean flow vs target magnitude",
    ))
    saved.extend(plot_metric(
        root,
        x,
        mrs,
        name="flow_target_magnitude_mrs",
        ylabel="MRS (lower is better)",
        title="Multi-task score vs target magnitude",
    ))
    saved.extend(plot_metric(
        root,
        x,
        mde,
        name="flow_target_magnitude_depth_rmse",
        ylabel="Depth RMSE to target (lower is better)",
        title="Depth target error vs flow target magnitude",
    ))
    saved.extend(plot_metric(
        root,
        x,
        iou,
        name="flow_target_magnitude_seg_iou",
        ylabel="Target IoU (higher is better)",
        title="Segmentation target IoU vs flow target magnitude",
    ))
    saved.extend(plot_metric_by_angle(
        root,
        rows,
        "aee_target_attack",
        name="flow_target_magnitude_direction_target_aee",
        ylabel="Flow target AEE (lower is better)",
        title="Targeted flow success vs magnitude and direction",
    ))
    saved.extend(plot_metric_by_angle(
        root,
        rows,
        "aee_init_attack",
        name="flow_target_magnitude_direction_delta_aee",
        ylabel="AEE clean vs attacked (higher is stronger change)",
        title="Change from clean flow vs magnitude and direction",
    ))
    saved.extend(plot_metric_by_angle(
        root,
        rows,
        "multitask_robustness_score",
        name="flow_target_magnitude_direction_mrs",
        ylabel="MRS (lower is better)",
        title="Multi-task score vs magnitude and direction",
    ))
    saved.extend(plot_heatmap(
        root,
        rows,
        "aee_target_attack",
        name="flow_target_magnitude_direction_target_aee_heatmap",
        label="Flow target AEE",
        title="Target AEE over flow target magnitude and direction",
    ))
    saved.extend(plot_heatmap(
        root,
        rows,
        "aee_init_attack",
        name="flow_target_magnitude_direction_delta_aee_heatmap",
        label="AEE clean vs attacked",
        title="Clean-flow change over target magnitude and direction",
    ))
    saved.extend(plot_heatmap(
        root,
        rows,
        "multitask_robustness_score",
        name="flow_target_magnitude_direction_mrs_heatmap",
        label="MRS",
        title="MRS over flow target magnitude and direction",
    ))
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Sweep output root with mag_*.log files")
    args = parser.parse_args()

    rows = collect(args.root)
    if not rows:
        raise SystemExit(f"No mag_*.log files found in {args.root}")

    csv_path = write_csv(args.root, rows)
    plot_paths = maybe_plot(args.root, rows)

    target_rows = [r for r in rows if math.isfinite(float(r["aee_target_attack"]))]
    best_target = min(target_rows, key=lambda r: float(r["aee_target_attack"])) if target_rows else None
    delta_rows = [r for r in rows if math.isfinite(float(r["aee_init_attack"]))]
    best_delta = max(delta_rows, key=lambda r: float(r["aee_init_attack"])) if delta_rows else None

    print(f"Wrote {csv_path}")
    for path in plot_paths:
        print(f"Wrote {path}")
    if best_target:
        print(
            "Best direction/magnitude by target AEE: "
            f"{best_target['flow_target_angle_deg']} deg, "
            f"{best_target['flow_target_magnitude']} "
            f"(AEE={float(best_target['aee_target_attack']):.4f})"
        )
    if best_delta:
        print(
            "Strongest clean-flow change by direction/magnitude: "
            f"{best_delta['flow_target_angle_deg']} deg, "
            f"{best_delta['flow_target_magnitude']} "
            f"(AEE={float(best_delta['aee_init_attack']):.4f})"
        )


if __name__ == "__main__":
    main()
