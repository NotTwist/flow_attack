#!/usr/bin/env python3
"""Summarize a patch flow-target angle sweep from run logs."""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
from pathlib import Path


os.environ.setdefault("MPLCONFIGDIR", f"/tmp/matplotlib-{os.environ.get('USER', 'codex')}")


RE_ANGLE = re.compile(r"angle_([-+]?\d+(?:p\d+)?)")
RE_AEE_TARGET = re.compile(r"AEE \(attacked vs target\):\s*(?P<v>[-+.\deE]+|N/A)")
RE_AEE_ATTACK = re.compile(r"AEE \(init vs attack\):\s*(?P<v>[-+.\deE]+|N/A)")
RE_MRS = re.compile(r"Multi-task target ratio .*:\s*(?P<v>[-+.\deE]+|N/A)")


def parse_float(value: str) -> float:
    if value == "N/A":
        return math.nan
    return float(value)


def last_match(pattern: re.Pattern[str], text: str) -> float:
    matches = list(pattern.finditer(text))
    if not matches:
        return math.nan
    return parse_float(matches[-1].group("v"))


def angle_from_log(path: Path, text: str) -> float:
    cli_match = re.search(r"--flow_target_angle_deg\s+([-+]?\d+(?:\.\d+)?)", text)
    if cli_match:
        return float(cli_match.group(1))

    name_match = RE_ANGLE.search(path.stem)
    if name_match:
        return float(name_match.group(1).replace("p", "."))

    raise ValueError(f"Could not infer angle for {path}")


def collect(root: Path) -> list[dict[str, float | str]]:
    rows = []
    for log_path in sorted(root.glob("angle_*.log")):
        text = log_path.read_text(encoding="utf-8", errors="replace")
        angle = angle_from_log(log_path, text)
        rows.append(
            {
                "angle_deg": angle,
                "aee_target_attack": last_match(RE_AEE_TARGET, text),
                "aee_init_attack": last_match(RE_AEE_ATTACK, text),
                "multitask_robustness_score": last_match(RE_MRS, text),
                "log_path": str(log_path),
            }
        )
    rows.sort(key=lambda r: float(r["angle_deg"]))
    return rows


def write_csv(root: Path, rows: list[dict[str, float | str]]) -> Path:
    csv_path = root / "flow_target_angle_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "angle_deg",
                "aee_target_attack",
                "aee_init_attack",
                "multitask_robustness_score",
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


def maybe_plot(root: Path, rows: list[dict[str, float | str]]) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:
        print(f"Skipping plots: matplotlib/numpy unavailable ({exc})")
        return []

    angles = np.array([float(r["angle_deg"]) for r in rows], dtype=float)
    target_aee = np.array([float(r["aee_target_attack"]) for r in rows], dtype=float)
    delta_aee = np.array([float(r["aee_init_attack"]) for r in rows], dtype=float)
    mrs = np.array([float(r["multitask_robustness_score"]) for r in rows], dtype=float)

    saved = []
    line_path = root / "flow_target_angle_summary.png"
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(angles, target_aee, marker="o", label="Flow target AEE")
    ax1.set_xlabel("Target angle, degrees (0=right, 90=down)")
    ax1.set_ylabel("Flow target AEE (lower is better)")
    ax1.grid(True, alpha=0.3)
    if np.isfinite(mrs).any():
        ax2 = ax1.twinx()
        ax2.plot(angles, mrs, marker="s", color="tab:orange", label="MRS")
        ax2.set_ylabel("MRS (lower is better)")
    fig.tight_layout()
    saved.extend(save_figure(fig, line_path))
    plt.close(fig)

    finite = np.isfinite(target_aee)
    if finite.any():
        polar_path = root / "flow_target_angle_polar.png"
        theta = np.deg2rad(angles[finite])
        values = target_aee[finite]
        fig = plt.figure(figsize=(5.5, 5.5))
        ax = fig.add_subplot(111, projection="polar")
        ax.plot(theta, values, marker="o")
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(-1)
        ax.set_title("Flow target AEE by target direction")
        fig.tight_layout()
        saved.extend(save_figure(fig, polar_path))
        plt.close(fig)

    finite_delta = np.isfinite(delta_aee)
    if finite_delta.any():
        delta_path = root / "flow_delta_aee_by_angle.png"
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(angles[finite_delta], delta_aee[finite_delta], marker="o", color="tab:green")
        ax.set_xlabel("Target angle, degrees (0=right, 90=down)")
        ax.set_ylabel("AEE clean vs attacked (higher is stronger change)")
        ax.set_title("Change from clean flow by target direction")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        saved.extend(save_figure(fig, delta_path))
        plt.close(fig)

        delta_polar_path = root / "flow_delta_aee_by_angle_polar.png"
        theta = np.deg2rad(angles[finite_delta])
        values = delta_aee[finite_delta]
        fig = plt.figure(figsize=(5.5, 5.5))
        ax = fig.add_subplot(111, projection="polar")
        ax.plot(theta, values, marker="o", color="tab:green")
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(-1)
        ax.set_title("Clean-vs-attacked AEE by target direction")
        fig.tight_layout()
        saved.extend(save_figure(fig, delta_polar_path))
        plt.close(fig)

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Sweep output root with angle_*.log files")
    args = parser.parse_args()

    rows = collect(args.root)
    if not rows:
        raise SystemExit(f"No angle_*.log files found in {args.root}")

    csv_path = write_csv(args.root, rows)
    plot_paths = maybe_plot(args.root, rows)

    flow_rows = [r for r in rows if math.isfinite(float(r["aee_target_attack"]))]
    best_flow = min(flow_rows, key=lambda r: float(r["aee_target_attack"])) if flow_rows else None

    print(f"Wrote {csv_path}")
    for path in plot_paths:
        print(f"Wrote {path}")
    if best_flow:
        print(
            "Best flow-target angle by AEE: "
            f"{best_flow['angle_deg']} deg "
            f"(AEE={float(best_flow['aee_target_attack']):.4f})"
        )
    delta_rows = [r for r in rows if math.isfinite(float(r["aee_init_attack"]))]
    best_delta = max(delta_rows, key=lambda r: float(r["aee_init_attack"])) if delta_rows else None
    if best_delta:
        print(
            "Strongest clean-flow change by AEE: "
            f"{best_delta['angle_deg']} deg "
            f"(AEE={float(best_delta['aee_init_attack']):.4f})"
        )


if __name__ == "__main__":
    main()
