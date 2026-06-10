#!/usr/bin/env python3
"""Summarize MDE per-image vs dataset-global target ablation logs."""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
from pathlib import Path


os.environ.setdefault("MPLCONFIGDIR", f"/tmp/matplotlib-{os.environ.get('USER', 'codex')}")


RE_AEE_TARGET = re.compile(r"AEE \(attacked vs target\):\s*(?P<v>[-+.\deE]+|N/A)")
RE_MRS = re.compile(r"Multi-task target ratio .*:\s*(?P<v>[-+.\deE]+|N/A)")
RE_MDE_ATTACK = re.compile(r"MDE RMSE \(attacked vs target\):\s*(?P<v>[-+.\deE]+|N/A)")
RE_MDE_CLEAN = re.compile(r"MDE RMSE \(clean vs target\):\s*(?P<v>[-+.\deE]+|N/A)")
RE_IOU_TARGET = re.compile(r"IoU \(attacked vs target\):\s*(?P<v>[-+.\deE]+|N/A)")


def parse_float(value: str) -> float:
    return math.nan if value == "N/A" else float(value)


def last_match(pattern: re.Pattern[str], text: str) -> float:
    matches = list(pattern.finditer(text))
    if not matches:
        return math.nan
    return parse_float(matches[-1].group("v"))


def target_kind(name: str) -> str:
    return "near" if name.startswith("near") else "far" if name.startswith("far") else name


def target_scope(name: str) -> str:
    return "global" if name.endswith("_global") else "per_image"


def collect(root: Path) -> list[dict[str, float | str]]:
    rows = []
    for log_path in sorted(root.glob("*.log")):
        name = log_path.stem
        if name not in {"near", "near_global", "far", "far_global"}:
            continue
        text = log_path.read_text(encoding="utf-8", errors="replace")
        rows.append(
            {
                "mde_target": name,
                "kind": target_kind(name),
                "scope": target_scope(name),
                "mde_rmse_target_attack": last_match(RE_MDE_ATTACK, text),
                "mde_rmse_clean_target": last_match(RE_MDE_CLEAN, text),
                "multitask_robustness_score": last_match(RE_MRS, text),
                "aee_target_attack": last_match(RE_AEE_TARGET, text),
                "iou_target_attack": last_match(RE_IOU_TARGET, text),
                "log_path": str(log_path),
            }
        )
    order = {"near": 0, "near_global": 1, "far": 2, "far_global": 3}
    rows.sort(key=lambda r: order.get(str(r["mde_target"]), 99))
    return rows


def write_csv(rows: list[dict[str, float | str]], root: Path) -> Path:
    path = root / "mde_target_scope_summary.csv"
    fieldnames = [
        "mde_target",
        "kind",
        "scope",
        "mde_rmse_target_attack",
        "mde_rmse_clean_target",
        "multitask_robustness_score",
        "aee_target_attack",
        "iou_target_attack",
        "log_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def save_figure(fig, png_path: Path) -> list[Path]:
    pdf_path = png_path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=180)
    fig.savefig(pdf_path)
    return [png_path, pdf_path]


def maybe_plot(rows: list[dict[str, float | str]], root: Path) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:
        print(f"Skipping plots: matplotlib/numpy unavailable ({exc})")
        return []

    ok_rows = [r for r in rows if math.isfinite(float(r["mde_rmse_target_attack"]))]
    if not ok_rows:
        return []

    labels = [str(r["mde_target"]) for r in ok_rows]
    x = np.arange(len(labels))
    mde = np.array([float(r["mde_rmse_target_attack"]) for r in ok_rows])
    mrs = np.array([float(r["multitask_robustness_score"]) for r in ok_rows])

    saved = []
    path = root / "mde_target_scope_rmse.png"
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["tab:blue" if r["scope"] == "per_image" else "tab:orange" for r in ok_rows]
    ax.bar(x, mde, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Depth RMSE to target (lower is better)")
    ax.set_title("MDE target construction ablation")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    saved.extend(save_figure(fig, path))
    plt.close(fig)

    if np.isfinite(mrs).any():
        path = root / "mde_target_scope_mrs.png"
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(x, mrs, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("MRS (lower is better)")
        ax.set_title("MDE target construction ablation")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        saved.extend(save_figure(fig, path))
        plt.close(fig)

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Ablation output root with near*.log/far*.log files")
    args = parser.parse_args()

    rows = collect(args.root)
    if not rows:
        raise SystemExit(f"No target-scope logs found in {args.root}")

    csv_path = write_csv(rows, args.root)
    plot_paths = maybe_plot(rows, args.root)

    candidates = [r for r in rows if math.isfinite(float(r["mde_rmse_target_attack"]))]
    best_mde = min(candidates, key=lambda r: float(r["mde_rmse_target_attack"])) if candidates else None

    print(f"Wrote {csv_path}")
    for path in plot_paths:
        print(f"Wrote {path}")
    if best_mde:
        print(
            "Best MDE target by RMSE: "
            f"{best_mde['mde_target']} "
            f"(RMSE={float(best_mde['mde_rmse_target_attack']):.4f})"
        )


if __name__ == "__main__":
    main()
