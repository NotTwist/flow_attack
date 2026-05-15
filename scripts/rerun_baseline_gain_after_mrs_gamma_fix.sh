#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Recompute the trained-vs-baseline size sweep after fixing
# mean_mrs_gamma_* metrics. Existing CSV rows are overwritten.
#
# Fast check:
#   SUBSET_SIZE=20 bash scripts/rerun_baseline_gain_after_mrs_gamma_fix.sh
#
# Full sweep:
#   bash scripts/rerun_baseline_gain_after_mrs_gamma_fix.sh

export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-file:${PWD}/mlruns}"
PYTHON="${PYTHON:-python3}"

OUTPUT_DIR="${OUTPUT_DIR:-experiment_data/patch_baseline_gain_kitti15_no_projection_mrs_gamma_fix}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-patch_baseline_gain_kitti15_no_projection_mrs_gamma_fix}"
RUNS_PER_KIND="${RUNS_PER_KIND:-1}"
MIN_SIZE="${MIN_SIZE:-50}"
MAX_SIZE="${MAX_SIZE:-300}"
STEP_SIZE="${STEP_SIZE:-50}"
EVAL_MODE="${EVAL_MODE:-testing}"
SUBSET_SIZE="${SUBSET_SIZE:-0}"

subset_args=()
if [[ "${SUBSET_SIZE}" != "0" ]]; then
  subset_args+=(--subset_size "${SUBSET_SIZE}")
fi

"${PYTHON}" scripts/eval_patch_size_sweep_kitti15_no_projection.py \
  --experiment_root experiment_data \
  --output_dir "${OUTPUT_DIR}" \
  --experiment_name "${EXPERIMENT_NAME}" \
  --runs_per_kind "${RUNS_PER_KIND}" \
  --include_baselines \
  --overwrite \
  --min_size "${MIN_SIZE}" \
  --max_size "${MAX_SIZE}" \
  --step_size "${STEP_SIZE}" \
  --eval_mode "${EVAL_MODE}" \
  "${subset_args[@]}" \
  "$@"
