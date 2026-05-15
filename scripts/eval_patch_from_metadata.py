#!/usr/bin/env python3
"""Evaluate a saved adversarial patch using checkpoint metadata.

The patch trainer writes a JSON file next to every checkpoint PNG. This helper
loads that JSON and reconstructs the run_patch_attack.py evaluation command.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


BOOL_FLAGS = {
    "attack_mde": "--attack_mde",
    "attack_ss": "--attack_ss",
    "patch_projection": "--patch_projection",
    "change_of_variables": "--change_of_variables",
    "normalize_losses": "--normalize_losses",
    "no_softmax": "--no_softmax",
    "plane_aug": "--plane_aug",
    "use_map_scaling": "--use_map_scaling",
}

VALUE_FLAGS = {
    "dataset": "--dataset",
    "eval_mode": "--eval_mode",
    "model_name": "--model_name",
    "mde_model": "--mde_model",
    "ss_model": "--ss_model",
    "target": "--target",
    "mde_target": "--mde_target",
    "ss_target": "--ss_target",
    "patch_size": "--patch_size",
    "y_scale": "--y_scale",
    "flow_shift": "--flow_shift",
    "flow_target_magnitude": "--flow_target_magnitude",
    "down_loss": "--down_loss",
    "down_hinge_min_mag_ratio": "--down_hinge_min_mag_ratio",
    "down_hinge_horizontal_weight": "--down_hinge_horizontal_weight",
    "down_hinge_magnitude_weight": "--down_hinge_magnitude_weight",
    "down_hinge_vertical_weight": "--down_hinge_vertical_weight",
    "mde_near_margin": "--mde_near_margin",
    "ss_focal_gamma": "--ss_focal_gamma",
    "weight_strategy": "--weight_strategy",
    "minmax_alpha_w": "--minmax_alpha_w",
    "minmax_gamma": "--minmax_gamma",
    "target_layer": "--target_layer",
    "scaling_type": "--scaling_type",
    "num_samples": "--num_samples",
    "tv_weight": "--tv_weight",
    "nps_weight": "--nps_weight",
    "defense": "--defense",
    "temp_window": "--temp_window",
    "sigma_color": "--sigma_color",
    "sigma_spatial": "--sigma_spatial",
    "k": "--k",
    "o": "--o",
    "t": "--t",
    "s": "--s",
    "r": "--r",
}


def resolve_metadata(metadata_arg: str | None, patch_arg: str | None) -> Path:
    if metadata_arg:
        return Path(metadata_arg)
    if not patch_arg:
        raise SystemExit("Provide --metadata or --patch")

    patch = Path(patch_arg)
    same_stem = patch.with_suffix(".json")
    if same_stem.exists():
        return same_stem

    latest_json = patch.parent / "latest.json"
    if patch.name == "latest.png" and latest_json.exists():
        return latest_json

    raise SystemExit(
        f"Could not find metadata for {patch}. Expected {same_stem}"
        f" or {latest_json}."
    )


def add_value(cmd: list[str], flag: str, value):
    if value is None:
        return
    cmd.extend([flag, str(value)])


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved patch checkpoint using metadata")
    parser.add_argument("--metadata", default="", help="Path to patch checkpoint JSON")
    parser.add_argument("--patch", default="", help="Path to patch PNG; metadata is inferred next to it")
    parser.add_argument("--output_dir", default="", help="Override evaluation output dir")
    parser.add_argument("--experiment_name", default="", help="Override MLflow experiment name")
    parser.add_argument("--eval_mode", default="", choices=["", "training", "testing"], help="Override eval split")
    parser.add_argument("--subset_size", type=int, default=-1, help="Override subset size; -1 keeps metadata value")
    parser.add_argument("--save_artifacts", action="store_true", help="Save evaluation artifacts")
    parser.add_argument("--eval_artifact_limit", type=int, default=0, help="Maximum number of eval batches to save artifacts for.")
    parser.add_argument("--dry_run", action="store_true", help="Print command without executing")
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Extra args appended after --, e.g. -- --ss_model pspnet_cityscapes",
    )
    cli = parser.parse_args()

    metadata_path = resolve_metadata(cli.metadata, cli.patch)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    train_args = dict(metadata.get("args", {}))
    patch_path = Path(cli.patch or metadata.get("patch_path", ""))
    if not patch_path.is_absolute():
        candidate = (metadata_path.parent / patch_path).resolve()
        patch_path = candidate if candidate.exists() else patch_path.resolve()

    run_id = metadata_path.stem
    output_dir = cli.output_dir or f"experiment_data/eval_{run_id}"
    experiment_name = cli.experiment_name or f"eval_{run_id}"

    cmd = [
        sys.executable,
        "run_patch_attack.py",
        "--trained_patch",
        str(patch_path),
        "--output_dir",
        output_dir,
        "--experiment_name",
        experiment_name,
    ]

    for key, flag in BOOL_FLAGS.items():
        if bool(train_args.get(key, False)):
            cmd.append(flag)

    # argparse uses --random_loc as store_false, so reproduce a non-random run.
    if train_args.get("random_loc") is False:
        cmd.append("--random_loc")

    for key, flag in VALUE_FLAGS.items():
        value = train_args.get(key)
        if key == "eval_mode" and cli.eval_mode:
            value = cli.eval_mode
        add_value(cmd, flag, value)

    loss_weights = train_args.get("loss_weights")
    if loss_weights is not None:
        cmd.append("--loss_weights")
        cmd.extend(str(v) for v in loss_weights)

    saved_iterations = train_args.get("saved_iterations") or []
    if saved_iterations:
        cmd.append("--saved_iterations")
        cmd.extend(str(v) for v in saved_iterations)

    if cli.subset_size >= 0:
        cmd.extend(["--subset_size", str(cli.subset_size)])
    elif int(train_args.get("subset_size", 0) or 0) > 0:
        cmd.extend(["--subset_size", str(train_args["subset_size"])])

    if cli.save_artifacts:
        cmd.append("--save_artifacts")
    if cli.eval_artifact_limit > 0:
        cmd.extend(["--eval_artifact_limit", str(cli.eval_artifact_limit)])

    extra_args = cli.extra_args
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    cmd.extend(extra_args)

    print("Loaded metadata:", metadata_path)
    print("Patch:", patch_path)
    print("Command:")
    print(" ".join(cmd))

    if cli.dry_run:
        return

    env = os.environ.copy()
    env.setdefault("MLFLOW_TRACKING_URI", f"file:{(Path.cwd() / 'mlruns').resolve()}")
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
