#!/usr/bin/env python3
"""Evaluate saved KITTI15 patch checkpoints at multiple rendered patch sizes.

The script is intentionally a wrapper around run_patch_attack.py, so evaluation
uses the same models, targets and metrics as the normal patch experiments. It
discovers the latest diffusion/pixel training runs, selects one checkpoint per
run, evaluates each checkpoint for a size sweep, then writes CSV, plots and a
short markdown trend summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-flow-attack")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from eval_patch_from_metadata import BOOL_FLAGS, VALUE_FLAGS, add_value  # noqa: E402


EXTRA_VALUE_FLAGS = {
    "patch_parametrization": "--patch_parametrization",
    "diffusion_base_image": "--diffusion_base_image",
    "diffusion_prompt": "--diffusion_prompt",
    "diffusion_init_mode": "--diffusion_init_mode",
}

METRICS = [
    "mean_multitask_robustness_score",
    "mean_aee_target_attack",
    "mean_aee_init_attack",
    "mean_mde_rmse_target_attack",
    "mean_mde_rmse_init_attack",
    "mean_ase_target_attack",
    "mean_ase_init_attack",
    "mean_iou_target_attack",
    "mean_iou_init_attack",
    "mean_mrs_gamma_flow",
    "mean_mrs_gamma_depth",
    "mean_mrs_gamma_seg",
]

PLOT_SPECS = [
    (
        "mean_multitask_robustness_score",
        "Mean Robustness Score vs Patch Size",
        "mean robustness score (lower is stronger)",
        "mean_robustness_score_vs_size.png",
    ),
    (
        "mean_aee_target_attack",
        "Flow Target AEE vs Patch Size",
        "AEE to target flow (lower is stronger)",
        "flow_target_aee_vs_size.png",
    ),
    (
        "mean_mde_rmse_target_attack",
        "Depth Target RMSE vs Patch Size",
        "RMSE to target depth (lower is stronger)",
        "depth_target_rmse_vs_size.png",
    ),
    (
        "mean_iou_target_attack",
        "Segmentation Target IoU vs Patch Size",
        "target IoU (higher is stronger)",
        "segmentation_target_iou_vs_size.png",
    ),
]

GAIN_SPECS = [
    (
        "mean_multitask_robustness_score",
        "Mean Robustness Score Gain vs Patch Size",
        "baseline - trained (positive = trained is stronger)",
        "gain_mean_robustness_score_vs_size.png",
        "lower",
    ),
    (
        "mean_aee_target_attack",
        "Flow Target AEE Gain vs Patch Size",
        "baseline - trained (positive = trained is stronger)",
        "gain_flow_target_aee_vs_size.png",
        "lower",
    ),
    (
        "mean_mde_rmse_target_attack",
        "Depth Target RMSE Gain vs Patch Size",
        "baseline - trained (positive = trained is stronger)",
        "gain_depth_target_rmse_vs_size.png",
        "lower",
    ),
    (
        "mean_iou_target_attack",
        "Segmentation Target IoU Gain vs Patch Size",
        "trained - baseline (positive = trained is stronger)",
        "gain_segmentation_target_iou_vs_size.png",
        "higher",
    ),
]

GAIN_METRIC_NAMES = {
    "mean_multitask_robustness_score": "mrs_gain",
    "mean_aee_target_attack": "flow_target_aee_gain",
    "mean_mde_rmse_target_attack": "depth_target_rmse_gain",
    "mean_iou_target_attack": "segmentation_target_iou_gain",
}


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def epoch_from_name(path: Path) -> int:
    match = re.search(r"patch_epoch_(\d+)", path.name)
    return int(match.group(1)) if match else -1


def step_from_metadata(metadata_path: Path) -> int:
    data = load_json(metadata_path)
    value = data.get("step")
    if value is not None:
        return int(value)
    batch = data.get("batch")
    if batch is not None:
        return int(batch)
    return epoch_from_name(metadata_path) * 1_000_000


def patch_exists(metadata_path: Path) -> bool:
    data = load_json(metadata_path)
    patch_path = Path(data.get("patch_path", ""))
    if patch_path.is_absolute():
        return patch_path.exists()
    if patch_path.exists():
        return True
    return (metadata_path.parent / patch_path).exists()


def resolve_patch_path(metadata_path: Path) -> Path:
    data = load_json(metadata_path)
    patch_path = Path(data.get("patch_path", ""))
    candidates = []
    if patch_path.is_absolute():
        candidates.append(patch_path)
    else:
        candidates.extend([REPO_ROOT / patch_path, metadata_path.parent / patch_path])
    candidates.append(metadata_path.with_suffix(".png"))

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Patch image not found for metadata {metadata_path}")


def is_completed_epoch(path: Path) -> bool:
    return re.fullmatch(r"patch_epoch_\d+\.json", path.name) is not None


def discover_runs(root: Path, prefix: str, limit: int) -> list[Path]:
    runs = [p for p in root.glob(f"{prefix}_*") if (p / "patch_checkpoints").is_dir()]
    runs.sort(key=lambda p: (p.stat().st_mtime, p.name), reverse=True)
    return runs[:limit]


def select_checkpoint(run_dir: Path, kind: str, allow_partial: bool = False) -> Path:
    checkpoint_dir = run_dir / "patch_checkpoints"
    metadata_files = sorted(checkpoint_dir.glob("patch_epoch_*.json"))
    metadata_files = [p for p in metadata_files if patch_exists(p)]
    if not metadata_files:
        raise FileNotFoundError(f"No patch checkpoint metadata found in {checkpoint_dir}")

    completed = [p for p in metadata_files if is_completed_epoch(p)]
    if completed:
        completed.sort(key=epoch_from_name)
        return completed[-1]

    if not allow_partial:
        raise FileNotFoundError(f"No completed epoch checkpoint found in {checkpoint_dir}")

    # If the latest training run is still partial, use the newest batch
    # checkpoint instead of failing. This keeps the script useful during long
    # experiments, but the label makes it clear that it is not a full epoch.
    metadata_files.sort(key=step_from_metadata)
    return metadata_files[-1]


def candidate_label(kind: str, run_dir: Path, metadata_path: Path) -> str:
    suffix = run_dir.name.replace(f"{kind}_patch_kitti15_no_projection_", "")
    return f"{kind}_{suffix}_{metadata_path.stem}"


def discover_candidates(args: argparse.Namespace) -> list[dict[str, Any]]:
    candidates = []
    specs = [
        ("diffusion", "diffusion_patch_kitti15_no_projection"),
        ("pixel", "pixel_patch_kitti15_no_projection"),
    ]
    for kind, prefix in specs:
        for run_dir in discover_runs(args.experiment_root, prefix, 10_000):
            if len([c for c in candidates if c["kind"] == kind]) >= args.runs_per_kind:
                break
            try:
                metadata_path = select_checkpoint(run_dir, kind, args.allow_partial_checkpoints)
            except FileNotFoundError as exc:
                print(f"Skipping {run_dir.name}: {exc}")
                continue
            patch_path = resolve_patch_path(metadata_path)
            metadata = load_json(metadata_path)
            candidates.append(
                {
                    "kind": kind,
                    "variant": "trained",
                    "paired_candidate": candidate_label(kind, run_dir, metadata_path),
                    "run_dir": run_dir,
                    "metadata_path": metadata_path,
                    "patch_path": patch_path,
                    "metadata": metadata,
                    "label": candidate_label(kind, run_dir, metadata_path),
                }
            )
    if args.include_baselines:
        paired = []
        for candidate in candidates:
            baseline = dict(candidate)
            baseline["variant"] = "baseline"
            baseline["label"] = f"{candidate['label']}_baseline"
            baseline["patch_path"] = ""
            paired.extend([candidate, baseline])
        return paired
    return candidates


def build_eval_command(
    candidate: dict[str, Any],
    patch_size: int,
    output_dir: Path,
    experiment_name: str,
    subset_size: int | None,
    eval_mode: str | None,
    extra_args: list[str],
) -> list[str]:
    train_args = dict(candidate["metadata"].get("args", {}))
    value_flags = dict(VALUE_FLAGS)
    value_flags.update(EXTRA_VALUE_FLAGS)
    value_flags.pop("patch_size", None)

    cmd = [sys.executable, "run_patch_attack.py"]
    if candidate.get("variant") == "baseline":
        cmd.append("--baseline")
    else:
        cmd.extend(["--trained_patch", str(candidate["patch_path"])])

    cmd.extend(
        [
            "--patch_size",
            str(patch_size),
            "--output_dir",
            str(output_dir),
            "--experiment_name",
            experiment_name,
        ]
    )

    for key, flag in BOOL_FLAGS.items():
        if bool(train_args.get(key, False)):
            cmd.append(flag)

    if train_args.get("random_loc") is False:
        cmd.append("--random_loc")

    for key, flag in value_flags.items():
        if key.startswith("diffusion_") and train_args.get("patch_parametrization") != "diffusion":
            continue
        value = train_args.get(key)
        if key == "eval_mode" and eval_mode:
            value = eval_mode
        if value == "":
            continue
        add_value(cmd, flag, value)

    loss_weights = train_args.get("loss_weights")
    if loss_weights is not None:
        cmd.append("--loss_weights")
        cmd.extend(str(v) for v in loss_weights)

    if subset_size is not None:
        cmd.extend(["--subset_size", str(subset_size)])
    elif int(train_args.get("subset_size", 0) or 0) > 0:
        cmd.extend(["--subset_size", str(train_args["subset_size"])])

    cmd.extend(extra_args)
    return cmd


def load_existing_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return [row for row in csv.DictReader(f) if row.get("mlflow_run_id") != "DRY_RUN"]


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "kind",
        "variant",
        "paired_candidate",
        "candidate",
        "run_dir",
        "metadata",
        "patch",
        "patch_size",
        "eval_output_dir",
        "mlflow_run_id",
        *METRICS,
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def pdf_name(filename: str) -> str:
    return str(Path(filename).with_suffix(".pdf"))


def save_plot(output_dir: Path, filename: str) -> None:
    plt.savefig(output_dir / filename, dpi=180)
    plt.savefig(output_dir / pdf_name(filename))


def fetch_mlflow_metrics(experiment_name: str, eval_output_dir: Path) -> tuple[str, dict[str, float]]:
    import mlflow
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment not found after eval: {experiment_name}")

    output_value = str(eval_output_dir)
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"params.output_dir = '{output_value}'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        # Fallback for absolute/relative path differences.
        all_runs = client.search_runs(
            [experiment.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=100,
        )
        runs = [r for r in all_runs if r.data.params.get("output_dir") == output_value]
    if not runs:
        raise RuntimeError(f"Could not find MLflow run for output_dir={output_value}")

    run = runs[0]
    return run.info.run_id, {name: safe_float(run.data.metrics.get(name)) for name in METRICS}


def run_one(
    candidate: dict[str, Any],
    patch_size: int,
    args: argparse.Namespace,
    extra_args: list[str],
) -> dict[str, Any]:
    eval_output_dir = args.output_dir / candidate["label"] / f"size_{patch_size:03d}"
    cmd = build_eval_command(
        candidate,
        patch_size,
        eval_output_dir,
        args.experiment_name,
        args.subset_size,
        args.eval_mode,
        extra_args,
    )

    print("\n==>", candidate["label"], "size", patch_size)
    print(" ".join(cmd))
    if args.dry_run:
        return {
            "kind": candidate["kind"],
            "variant": candidate.get("variant", "trained"),
            "paired_candidate": candidate.get("paired_candidate", candidate["label"]),
            "candidate": candidate["label"],
            "run_dir": str(candidate["run_dir"]),
            "metadata": str(candidate["metadata_path"]),
            "patch": str(candidate["patch_path"]),
            "patch_size": patch_size,
            "eval_output_dir": str(eval_output_dir),
            "mlflow_run_id": "DRY_RUN",
        }

    env = os.environ.copy()
    env.setdefault("MLFLOW_TRACKING_URI", f"file:{(REPO_ROOT / 'mlruns').resolve()}")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True, env=env)

    run_id, metrics = fetch_mlflow_metrics(args.experiment_name, eval_output_dir)
    row = {
        "kind": candidate["kind"],
        "variant": candidate.get("variant", "trained"),
        "paired_candidate": candidate.get("paired_candidate", candidate["label"]),
        "candidate": candidate["label"],
        "run_dir": str(candidate["run_dir"]),
        "metadata": str(candidate["metadata_path"]),
        "patch": str(candidate["patch_path"]),
        "patch_size": patch_size,
        "eval_output_dir": str(eval_output_dir),
        "mlflow_run_id": run_id,
    }
    row.update(metrics)
    return row


def grouped_numeric(rows: list[dict[str, Any]], metric: str) -> dict[str, list[tuple[int, float]]]:
    grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        if row.get("variant", "trained") == "baseline":
            continue
        value = safe_float(row.get(metric))
        size = int(row["patch_size"])
        if math.isfinite(value):
            grouped[display_label(str(row["candidate"]))].append((size, value))
    for values in grouped.values():
        values.sort()
    return grouped


def display_label(label: str) -> str:
    """Human-readable label for plots and summaries."""
    is_baseline = label.endswith("_baseline")
    base_label = label[:-len("_baseline")] if is_baseline else label

    if base_label.startswith("diffusion_"):
        return "Diffusion baseline" if is_baseline else "Diffusion patch"
    if base_label.startswith("pixel_"):
        return "Random noise baseline" if is_baseline else "Pixel patch"
    return label


def gain_rows(rows: list[dict[str, Any]], metric: str, direction: str) -> dict[str, list[tuple[int, float]]]:
    trained: dict[tuple[str, int], float] = {}
    baseline: dict[tuple[str, int], float] = {}
    for row in rows:
        pair = str(row.get("paired_candidate") or row.get("candidate"))
        size = int(row["patch_size"])
        value = safe_float(row.get(metric))
        if not math.isfinite(value):
            continue
        variant = row.get("variant", "trained")
        key = (pair, size)
        if variant == "baseline":
            baseline[key] = value
        else:
            trained[key] = value

    grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for key, trained_value in trained.items():
        if key not in baseline:
            continue
        pair, size = key
        baseline_value = baseline[key]
        gain = baseline_value - trained_value if direction == "lower" else trained_value - baseline_value
        grouped[display_label(pair)].append((size, gain))

    for values in grouped.values():
        values.sort()
    return grouped


def collect_gain_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pair_sizes = sorted(
        {
            (str(row.get("paired_candidate") or row.get("candidate")), int(row["patch_size"]))
            for row in rows
            if row.get("variant", "trained") != "baseline" and row.get("patch_size")
        }
    )
    out_rows: list[dict[str, Any]] = []
    for pair, size in pair_sizes:
        out_row: dict[str, Any] = {"paired_candidate": pair, "patch_size": size}
        has_gain = False
        for metric, _title, _ylabel, _filename, direction in GAIN_SPECS:
            values = dict(gain_rows(rows, metric, direction).get(display_label(pair), []))
            gain = values.get(size, math.nan)
            out_row[GAIN_METRIC_NAMES[metric]] = gain
            has_gain = has_gain or math.isfinite(gain)
        if has_gain:
            out_rows.append(out_row)
    return out_rows


def write_gain_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "paired_candidate",
        "patch_size",
        "mrs_gain",
        "flow_target_aee_gain",
        "depth_target_rmse_gain",
        "segmentation_target_iou_gain",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def plot_metric(rows: list[dict[str, Any]], output_dir: Path, metric: str, title: str, ylabel: str, filename: str) -> None:
    grouped = grouped_numeric(rows, metric)
    if not grouped:
        return

    plt.figure(figsize=(10, 6))
    for label, values in grouped.items():
        sizes = [x for x, _ in values]
        ys = [y for _, y in values]
        plt.plot(sizes, ys, marker="o", linewidth=2, label=label)
    plt.title(title)
    plt.xlabel("patch size, px")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    save_plot(output_dir, filename)
    plt.close()


def plot_gain(rows: list[dict[str, Any]], output_dir: Path, metric: str, title: str, ylabel: str, filename: str, direction: str) -> None:
    grouped = gain_rows(rows, metric, direction)
    if not grouped:
        return

    plt.figure(figsize=(10, 6))
    for label, values in grouped.items():
        sizes = [x for x, _ in values]
        ys = [y for _, y in values]
        plt.plot(sizes, ys, marker="o", linewidth=2, label=label)
    plt.axhline(0.0, color="black", linewidth=1, alpha=0.5)
    plt.title(title)
    plt.xlabel("patch size, px")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    save_plot(output_dir, filename)
    plt.close()


def trend(values: list[tuple[int, float]]) -> str:
    if len(values) < 2:
        return "not enough points"
    first_size, first = values[0]
    last_size, last = values[-1]
    if not (math.isfinite(first) and math.isfinite(last)):
        return "contains non-finite values"
    delta = last - first
    rel = (delta / abs(first) * 100.0) if abs(first) > 1e-12 else math.nan
    direction = "decreases" if delta < 0 else "increases" if delta > 0 else "is unchanged"
    return f"{direction} from {first:.4g} at {first_size}px to {last:.4g} at {last_size}px ({rel:+.1f}%)"


def write_summary(rows: list[dict[str, Any]], output_dir: Path) -> None:
    lines = [
        "# Patch Size Sweep Summary",
        "",
        "Lower mean robustness score, Flow target AEE and depth RMSE mean a stronger targeted attack. "
        "For target IoU, higher is stronger.",
        "",
    ]

    mrs = grouped_numeric(rows, "mean_multitask_robustness_score")
    for label, values in sorted(mrs.items()):
        best_size, best_value = min(values, key=lambda item: item[1])
        lines.append(f"## {label}")
        lines.append("")
        lines.append(f"- Mean robustness score {trend(values)}.")
        lines.append(f"- Best size by mean robustness score: {best_size}px ({best_value:.4g}).")

        for metric, name in [
            ("mean_aee_target_attack", "Flow target AEE"),
            ("mean_mde_rmse_target_attack", "depth target RMSE"),
            ("mean_iou_target_attack", "segmentation target IoU"),
        ]:
            grouped = grouped_numeric(rows, metric)
            if label not in grouped:
                continue
            lines.append(f"- {name}: {trend(grouped[label])}.")
        lines.append("")

    output_path = output_dir / "summary.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_gain_summary(rows: list[dict[str, Any]], output_dir: Path) -> None:
    lines = [
        "# Baseline Gain Summary",
        "",
        "Positive gain means the trained patch is stronger than its baseline at the same rendered size.",
        "For lower-is-stronger metrics gain is `baseline - trained`; for target IoU gain is `trained - baseline`.",
        "",
    ]

    mrs_gain = gain_rows(rows, "mean_multitask_robustness_score", "lower")
    for label, values in sorted(mrs_gain.items()):
        best_size, best_value = max(values, key=lambda item: item[1])
        lines.append(f"## {label}")
        lines.append("")
        lines.append(f"- Mean robustness score gain {trend(values)}.")
        lines.append(f"- Best size by robustness gain: {best_size}px ({best_value:+.4g}).")
        for metric, name, direction in [
            ("mean_aee_target_attack", "Flow target AEE gain", "lower"),
            ("mean_mde_rmse_target_attack", "depth target RMSE gain", "lower"),
            ("mean_iou_target_attack", "target IoU gain", "higher"),
        ]:
            grouped = gain_rows(rows, metric, direction)
            if label not in grouped:
                continue
            lines.append(f"- {name}: {trend(grouped[label])}.")
        lines.append("")

    if len(lines) == 5:
        lines.append("No trained/baseline pairs found in the CSV yet.")

    (output_dir / "baseline_gain_summary.md").write_text("\n".join(lines), encoding="utf-8")


def make_plots_and_summary(rows: list[dict[str, Any]], output_dir: Path) -> None:
    if not rows:
        return
    for metric, title, ylabel, filename in PLOT_SPECS:
        plot_metric(rows, output_dir, metric, title, ylabel, filename)
    for metric, title, ylabel, filename, direction in GAIN_SPECS:
        plot_gain(rows, output_dir, metric, title, ylabel, filename, direction)
    write_gain_rows(output_dir / "baseline_gain_results.csv", collect_gain_table(rows))
    write_summary(rows, output_dir)
    write_gain_summary(rows, output_dir)


def parse_sizes(args: argparse.Namespace) -> list[int]:
    if args.sizes:
        return sorted(set(args.sizes))
    return list(range(args.min_size, args.max_size + 1, args.step_size))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment_root", type=Path, default=REPO_ROOT / "experiment_data")
    parser.add_argument("--output_dir", type=Path, default=REPO_ROOT / "experiment_data" / "patch_size_sweep_kitti15_no_projection")
    parser.add_argument("--experiment_name", default="patch_size_sweep_kitti15_no_projection")
    parser.add_argument("--runs_per_kind", type=int, default=1)
    parser.add_argument("--sizes", nargs="*", type=int, default=[])
    parser.add_argument("--min_size", type=int, default=50)
    parser.add_argument("--max_size", type=int, default=300)
    parser.add_argument("--step_size", type=int, default=50)
    parser.add_argument("--subset_size", type=int, default=None, help="Override eval subset. Omit for full dataset/metadata value.")
    parser.add_argument("--eval_mode", default="testing", choices=["training", "testing"])
    parser.add_argument("--overwrite", action="store_true", help="Re-run combinations already present in the CSV.")
    parser.add_argument("--dry_run", action="store_true", help="Print selected checkpoints and commands without running evaluation.")
    parser.add_argument("--plot_only", action="store_true", help="Only regenerate plots/summary from the existing CSV.")
    parser.add_argument("--allow_partial_checkpoints", action="store_true", help="Allow latest batch checkpoint when a run has no completed epoch.")
    parser.add_argument("--include_baselines", action="store_true", help="Also evaluate random-noise pixel and diffusion-base-image baselines.")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER, help="Extra args appended after --, e.g. -- --model_name raft")
    args = parser.parse_args()

    os.environ.setdefault("MLFLOW_TRACKING_URI", f"file:{(REPO_ROOT / 'mlruns').resolve()}")
    args.experiment_root = args.experiment_root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "patch_size_sweep_results.csv"
    rows: list[dict[str, Any]] = load_existing_rows(csv_path)

    if args.plot_only:
        make_plots_and_summary(rows, args.output_dir)
        print(f"Plots regenerated in {args.output_dir}")
        return

    extra_args = args.extra_args
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    candidates = discover_candidates(args)
    sizes = parse_sizes(args)
    print("Selected checkpoints:")
    for candidate in candidates:
        print(f"  {candidate['label']}")
        print(f"    variant:  {candidate.get('variant', 'trained')}")
        print(f"    metadata: {candidate['metadata_path']}")
        print(f"    patch:    {candidate['patch_path'] or 'baseline generated by run_patch_attack.py'}")
    print("Sizes:", sizes)

    completed = {(row["candidate"], int(row["patch_size"])) for row in rows if row.get("patch_size")}
    for candidate in candidates:
        for size in sizes:
            key = (candidate["label"], size)
            if key in completed and not args.overwrite:
                print(f"Skipping existing result: {candidate['label']} size {size}")
                continue
            row = run_one(candidate, size, args, extra_args)
            if args.dry_run:
                continue
            rows = [r for r in rows if not (r.get("candidate") == candidate["label"] and int(r.get("patch_size", -1)) == size)]
            rows.append(row)
            write_rows(csv_path, rows)

    if not args.dry_run:
        make_plots_and_summary(rows, args.output_dir)
        print(f"\nResults: {csv_path}")
        print(f"Plots and summary: {args.output_dir}")


if __name__ == "__main__":
    main()
