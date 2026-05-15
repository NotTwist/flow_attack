#!/usr/bin/env bash
set -euo pipefail

# Compares the latest trained diffusion/pixel patches with their baselines:
#   diffusion baseline = diffusion_base_image from the training metadata (dog.jpg)
#   pixel baseline     = fixed random-noise patch
#
# Extra arguments after "--" are forwarded to run_patch_attack.py.
# Example quick smoke:
#   bash scripts/eval_patch_baseline_gain_kitti15_no_projection.sh -- --subset_size 20

export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-file:${PWD}/mlruns}"

python scripts/eval_patch_size_sweep_kitti15_no_projection.py \
  --experiment_root experiment_data \
  --output_dir experiment_data/patch_baseline_gain_kitti15_no_projection \
  --experiment_name patch_baseline_gain_kitti15_no_projection \
  --runs_per_kind 1 \
  --include_baselines \
  --min_size 50 \
  --max_size 300 \
  --step_size 50 \
  --eval_mode testing \
  "$@"
