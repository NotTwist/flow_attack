#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Depth/segmentation-only interpretation of adv-manhole baseline.
# The runner still computes flow internally for the shared tracker, but the
# comparison should use only MDE/SS metrics:
#   mean_mde_rmse_init_attack, mean_ase_init_attack, mean_iou_init_attack
#
# Set TRAINED_PATCH to skip training and evaluate an existing adv-manhole PNG.

PYTHON="${PYTHON:-/home/28s_mur@lab.graphicon.ru/miniconda3/envs/attack/bin/python}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-file:${PWD}/mlruns}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-experiment_data/adv_manhole_kitti_depth_seg_${RUN_ID}}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-adv_manhole_kitti_depth_seg_baseline}"

SUBSET_SIZE="${SUBSET_SIZE:-0}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-1}"
TRAIN_TARGET_COVERAGE="${TRAIN_TARGET_COVERAGE:-0.015}"
EVAL_TARGET_COVERAGE="${EVAL_TARGET_COVERAGE:-0.015}"

trained_patch_args=()
if [[ -n "${TRAINED_PATCH:-}" ]]; then
  trained_patch_args+=(--trained_patch "${TRAINED_PATCH}")
fi
subset_args=()
if [[ "${SUBSET_SIZE}" != "0" ]]; then
  subset_args+=(--subset_size "${SUBSET_SIZE}")
fi

mkdir -p "${OUTPUT_DIR}/logs"

"${PYTHON}" run_adv_manhole_kitti.py \
  "${trained_patch_args[@]}" \
  "${subset_args[@]}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --flow_model raft \
  --mde_model depth-anything-v2 \
  --ss_model segformer_cityscapes \
  --eval_mode testing \
  --train_target_coverage "${TRAIN_TARGET_COVERAGE}" \
  --eval_target_coverage "${EVAL_TARGET_COVERAGE}" \
  --output_dir "${OUTPUT_DIR}" \
  --experiment_name "${EXPERIMENT_NAME}" \
  2>&1 | tee "${OUTPUT_DIR}/logs/eval.log"

echo "Adv-manhole depth/seg baseline finished."
echo "Use depth/seg metrics only in the diploma comparison."
echo "Output dir: ${OUTPUT_DIR}"
