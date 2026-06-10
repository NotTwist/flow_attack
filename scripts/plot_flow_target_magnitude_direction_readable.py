#!/usr/bin/env python3
"""Build readable plots for flow target magnitude/direction ablation."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", f"/tmp/matplotlib-{os.environ.get('USER', 'codex')}")


def load_rows(csv_path: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            parsed: dict[str, float | str] = {}
            for key, value in row.items():
                if key == "log_path":
                    parsed[key] = value
                else:
                    parsed[key] = float(value)
            rows.append(parsed)
    return rows


def save(fig, out: Path) -> list[Path]:
    png = out.with_suffix(".png")
    pdf = out.with_suffix(".pdf")
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    return [png, pdf]


def grid_for(rows, metric: str, angles: list[float], magnitudes: list[float]):
    import numpy as np

    grid = np.full((len(angles), len(magnitudes)), np.nan)
    index = {(float(r["flow_target_angle_deg"]), float(r["flow_target_magnitude"])): r for r in rows}
    for i, angle in enumerate(angles):
        for j, mag in enumerate(magnitudes):
            row = index.get((angle, mag))
            if row is not None:
                grid[i, j] = float(row[metric])
    return grid


def annotate_heatmap(ax, grid, *, lower_is_better: bool):
    import numpy as np

    finite = np.isfinite(grid)
    if not finite.any():
        return
    best = np.nanmin(grid) if lower_is_better else np.nanmax(grid)
    worst = np.nanmax(grid) if lower_is_better else np.nanmin(grid)
    span = max(abs(worst - best), 1e-9)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            value = grid[i, j]
            if not np.isfinite(value):
                continue
            text_color = "white" if abs(value - worst) / span < 0.55 else "black"
            weight = "bold" if math.isclose(value, best, rel_tol=1e-9, abs_tol=1e-9) else "normal"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7, color=text_color, fontweight=weight)


def plot_heatmaps(root: Path, rows, angles, magnitudes) -> list[Path]:
    import matplotlib.pyplot as plt
    import numpy as np

    specs = [
        ("aee_target_attack", "Flow target AEE", "lower is better", True, "viridis_r"),
        ("aee_init_attack", "AEE: clean vs attacked", "higher is stronger", False, "magma"),
        ("multitask_robustness_score", "MRS", "lower is better", True, "viridis_r"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), constrained_layout=True)
    for ax, (metric, title, subtitle, lower, cmap) in zip(axes, specs):
        grid = grid_for(rows, metric, angles, magnitudes)
        im = ax.imshow(np.ma.masked_invalid(grid), aspect="auto", origin="lower", cmap=cmap)
        ax.set_title(f"{title}\n{subtitle}", fontsize=11)
        ax.set_xticks(range(len(magnitudes)))
        ax.set_xticklabels([f"{m:g}" for m in magnitudes], rotation=35, ha="right")
        ax.set_yticks(range(len(angles)))
        ax.set_yticklabels([f"{a:g}" for a in angles])
        ax.set_xlabel("Target magnitude, px")
        ax.set_ylabel("Target angle, deg")
        annotate_heatmap(ax, grid, lower_is_better=lower)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return save(fig, root / "flow_target_magnitude_direction_heatmaps_readable")


def average_by(rows, group_key: str, metric: str):
    groups: dict[float, list[float]] = {}
    for row in rows:
        key = float(row[group_key])
        value = float(row[metric])
        if math.isfinite(value):
            groups.setdefault(key, []).append(value)
    xs = sorted(groups)
    ys = [sum(groups[x]) / len(groups[x]) for x in xs]
    return xs, ys


def plot_profiles(root: Path, rows) -> list[Path]:
    import matplotlib.pyplot as plt

    specs = [
        ("aee_target_attack", "Flow target AEE", "lower is better", "tab:blue"),
        ("aee_init_attack", "AEE clean vs attacked", "higher is stronger", "tab:orange"),
        ("multitask_robustness_score", "MRS", "lower is better", "tab:green"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.2), constrained_layout=True)
    for col, (metric, label, subtitle, color) in enumerate(specs):
        xs, ys = average_by(rows, "flow_target_magnitude", metric)
        axes[0, col].plot(xs, ys, marker="o", color=color)
        axes[0, col].set_xscale("log")
        axes[0, col].set_title(f"{label}\naveraged over directions, {subtitle}", fontsize=10)
        axes[0, col].set_xlabel("Target magnitude, px")
        axes[0, col].grid(True, alpha=0.3)

        xs, ys = average_by(rows, "flow_target_angle_deg", metric)
        axes[1, col].plot(xs, ys, marker="o", color=color)
        axes[1, col].set_title(f"{label}\naveraged over magnitudes, {subtitle}", fontsize=10)
        axes[1, col].set_xlabel("Target angle, deg")
        axes[1, col].set_xticks(xs)
        axes[1, col].grid(True, alpha=0.3)
    return save(fig, root / "flow_target_magnitude_direction_profiles_readable")


def plot_tradeoff(root: Path, rows) -> list[Path]:
    import matplotlib.pyplot as plt

    best_mrs = min(rows, key=lambda r: float(r["multitask_robustness_score"]))
    best_delta = max(rows, key=lambda r: float(r["aee_init_attack"]))
    best_target = min(rows, key=lambda r: float(r["aee_target_attack"]))

    fig, ax = plt.subplots(figsize=(7.6, 5.6), constrained_layout=True)
    sc = ax.scatter(
        [float(r["aee_target_attack"]) for r in rows],
        [float(r["aee_init_attack"]) for r in rows],
        c=[float(r["multitask_robustness_score"]) for r in rows],
        s=[28 + 5 * math.sqrt(float(r["flow_target_magnitude"])) for r in rows],
        cmap="viridis_r",
        edgecolor="black",
        linewidth=0.3,
        alpha=0.9,
    )
    for label, row in [("best target", best_target), ("best MRS", best_mrs), ("best delta", best_delta)]:
        x = float(row["aee_target_attack"])
        y = float(row["aee_init_attack"])
        ax.scatter([x], [y], s=110, facecolor="none", edgecolor="red", linewidth=1.8)
        ax.annotate(
            f"{label}\n{float(row['flow_target_angle_deg']):g} deg, m={float(row['flow_target_magnitude']):g}",
            (x, y),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("Flow target AEE (lower is closer to target)")
    ax.set_ylabel("AEE clean vs attacked (higher is stronger change)")
    ax.set_title("Target accuracy vs clean-flow change")
    ax.grid(True, alpha=0.3)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("MRS (lower is better)")
    return save(fig, root / "flow_target_magnitude_direction_tradeoff_readable")


def write_report(root: Path, rows, angles, magnitudes) -> Path:
    report = root / "flow_target_magnitude_direction_check.md"
    expected = len(angles) * len(magnitudes)
    missing = []
    present = {(float(r["flow_target_angle_deg"]), float(r["flow_target_magnitude"])) for r in rows}
    for angle in angles:
        for mag in magnitudes:
            if (angle, mag) not in present:
                missing.append((angle, mag))

    best_target = min(rows, key=lambda r: float(r["aee_target_attack"]))
    best_delta = max(rows, key=lambda r: float(r["aee_init_attack"]))
    best_mrs = min(rows, key=lambda r: float(r["multitask_robustness_score"]))
    constrained = [r for r in rows if float(r["flow_target_magnitude"]) >= 1.0]
    best_mrs_constrained = min(constrained, key=lambda r: float(r["multitask_robustness_score"]))

    def fmt(row, metric):
        return (
            f"angle={float(row['flow_target_angle_deg']):g} deg, "
            f"magnitude={float(row['flow_target_magnitude']):g}, "
            f"{metric}={float(row[metric]):.4f}, "
            f"delta={float(row['aee_init_attack']):.4f}, "
            f"MRS={float(row['multitask_robustness_score']):.4f}"
        )

    text = [
        "# Flow Target Magnitude/Direction Ablation Check",
        "",
        f"- Rows: {len(rows)}",
        f"- Expected grid: {len(angles)} angles x {len(magnitudes)} magnitudes = {expected}",
        f"- Missing pairs: {missing if missing else 'none'}",
        "",
        "## Best Points",
        "",
        f"- Best raw target AEE: {fmt(best_target, 'aee_target_attack')}",
        f"- Strongest clean-flow change: {fmt(best_delta, 'aee_init_attack')}",
        f"- Best MRS: {fmt(best_mrs, 'multitask_robustness_score')}",
        f"- Best MRS with magnitude >= 1 px: {fmt(best_mrs_constrained, 'multitask_robustness_score')}",
        "",
        "Note: raw target AEE is biased toward very small target magnitudes, because the target vector itself is near zero.",
    ]
    report.write_text("\n".join(text) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()

    csv_path = args.root / "flow_target_magnitude_summary.csv"
    rows = load_rows(csv_path)
    angles = sorted({float(r["flow_target_angle_deg"]) for r in rows})
    magnitudes = sorted({float(r["flow_target_magnitude"]) for r in rows})

    saved = []
    saved.extend(plot_heatmaps(args.root, rows, angles, magnitudes))
    saved.extend(plot_profiles(args.root, rows))
    saved.extend(plot_tradeoff(args.root, rows))
    report = write_report(args.root, rows, angles, magnitudes)

    for path in saved:
        print(f"Wrote {path}")
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()
