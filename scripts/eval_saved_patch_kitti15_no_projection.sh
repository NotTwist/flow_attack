#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Evaluate a saved patch PNG without retraining and without surface projection.
#
# Example:
#   TRAINED_PATCH=experiment_data/.../patch_checkpoints/latest.png \
#     bash scripts/eval_saved_patch_kitti15_no_projection.sh

if [[ -z "${TRAINED_PATCH:-}" ]]; then
  echo "ERROR: set TRAINED_PATCH=/path/to/patch.png" >&2
  exit 1
fi

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-experiment_data/eval_saved_patch_kitti15_no_projection_${RUN_ID}}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-eval_saved_patch_kitti15_no_projection_${RUN_ID}}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-file:${PWD}/mlruns}"

DATASET="${DATASET:-Kitti15}"
EVAL_MODE="${EVAL_MODE:-testing}"
FLOW_MODEL="${FLOW_MODEL:-raft}"
MDE_MODEL="${MDE_MODEL:-depth-anything-v2}"
SS_MODEL="${SS_MODEL:-segformer_cityscapes}"
PATCH_SIZE="${PATCH_SIZE:-300}"

FLOW_W="${FLOW_W:-1.0}"
MDE_W="${MDE_W:-0.1}"
SS_W="${SS_W:-1.0}"
FLOW_TARGET="${FLOW_TARGET:-down}"
MDE_TARGET="${MDE_TARGET:-near}"
SS_TARGET="${SS_TARGET:-targeted}"
FLOW_TARGET_MAGNITUDE="${FLOW_TARGET_MAGNITUDE:-10.0}"
MDE_NEAR_MARGIN="${MDE_NEAR_MARGIN:-0.1}"
SS_FOCAL_GAMMA="${SS_FOCAL_GAMMA:-2.0}"
TV_WEIGHT="${TV_WEIGHT:-0.01}"
NPS_WEIGHT="${NPS_WEIGHT:-0.1}"

SUBSET_SIZE="${SUBSET_SIZE:-0}"
SAVE_ARTIFACTS="${SAVE_ARTIFACTS:-1}"

subset_args=()
if [[ "${SUBSET_SIZE}" != "0" ]]; then
  subset_args+=(--subset_size "${SUBSET_SIZE}")
fi

artifact_args=()
if [[ "${SAVE_ARTIFACTS}" == "1" ]]; then
  artifact_args+=(--save_artifacts)
fi

mkdir -p "${OUTPUT_DIR}"

echo "============================================================"
echo "Evaluating saved patch on KITTI15 without projection"
echo "Patch:      ${TRAINED_PATCH}"
echo "Output dir: ${OUTPUT_DIR}"
echo "MLflow URI: ${MLFLOW_TRACKING_URI}"
echo "Physical regs: tv=${TV_WEIGHT}, nps=${NPS_WEIGHT}"
echo "============================================================"

python3 run_patch_attack.py \
  --dataset "${DATASET}" \
  --eval_mode "${EVAL_MODE}" \
  --model_name "${FLOW_MODEL}" \
  --mde_model "${MDE_MODEL}" \
  --ss_model "${SS_MODEL}" \
  --attack_mde \
  --attack_ss \
  --trained_patch "${TRAINED_PATCH}" \
  --patch_size "${PATCH_SIZE}" \
  --loss_weights "${FLOW_W}" "${MDE_W}" "${SS_W}" \
  --target "${FLOW_TARGET}" \
  --mde_target "${MDE_TARGET}" \
  --mde_near_margin "${MDE_NEAR_MARGIN}" \
  --ss_target "${SS_TARGET}" \
  --ss_focal_gamma "${SS_FOCAL_GAMMA}" \
  --flow_target_magnitude "${FLOW_TARGET_MAGNITUDE}" \
  --tv_weight "${TV_WEIGHT}" \
  --nps_weight "${NPS_WEIGHT}" \
  --output_dir "${OUTPUT_DIR}" \
  --experiment_name "${EXPERIMENT_NAME}" \
  "${subset_args[@]}" \
  "${artifact_args[@]}"
