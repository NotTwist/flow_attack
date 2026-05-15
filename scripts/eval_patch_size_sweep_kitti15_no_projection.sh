#!/usr/bin/env bash
set -euo pipefail

# Evaluates the latest diffusion and latest pixel patch runs on KITTI15
# for patch sizes 50, 100, 150, 200, 250 and 300 px.
#
# Extra arguments after "--" are forwarded to run_patch_attack.py.
# Example:
#   bash scripts/eval_patch_size_sweep_kitti15_no_projection.sh -- --subset_size 40

export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-file:${PWD}/mlruns}"

python scripts/eval_patch_size_sweep_kitti15_no_projection.py \
  --experiment_root experiment_data \
  --output_dir experiment_data/patch_size_sweep_kitti15_no_projection \
  --experiment_name patch_size_sweep_kitti15_no_projection \
  --runs_per_kind 1 \
  --min_size 50 \
  --max_size 300 \
  --step_size 50 \
  --eval_mode testing \
  "$@"
