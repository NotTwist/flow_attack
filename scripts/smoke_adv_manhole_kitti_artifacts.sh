#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Small visual check for the adv-manhole KITTI integration.
# It trains briefly, evaluates a few samples, and saves clean/attacked/mask
# artifacts so the rendered manhole patch can be inspected.

PYTHON="${PYTHON:-/home/28s_mur@lab.graphicon.ru/miniconda3/envs/attack/bin/python}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-file:${PWD}/mlruns}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-experiment_data/adv_manhole_kitti_visual_smoke_${RUN_ID}}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-adv_manhole_kitti_visual_smoke}"

SUBSET_SIZE="${SUBSET_SIZE:-4}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
EVAL_ARTIFACT_LIMIT="${EVAL_ARTIFACT_LIMIT:-3}"
TRAIN_TARGET_COVERAGE="${TRAIN_TARGET_COVERAGE:-0.015}"
EVAL_TARGET_COVERAGE="${EVAL_TARGET_COVERAGE:-0.015}"

mkdir -p "${OUTPUT_DIR}/logs"

"${PYTHON}" run_adv_manhole_kitti.py \
  --subset_size "${SUBSET_SIZE}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --flow_model raft \
  --mde_model depth-anything-v2 \
  --ss_model segformer_cityscapes \
  --eval_mode testing \
  --train_target_coverage "${TRAIN_TARGET_COVERAGE}" \
  --eval_target_coverage "${EVAL_TARGET_COVERAGE}" \
  --save_artifacts \
  --eval_artifact_limit "${EVAL_ARTIFACT_LIMIT}" \
  --output_dir "${OUTPUT_DIR}" \
  --experiment_name "${EXPERIMENT_NAME}" \
  2>&1 | tee "${OUTPUT_DIR}/logs/smoke.log"

echo "Visual smoke finished."
echo "Output dir: ${OUTPUT_DIR}"
echo "Patch:      ${OUTPUT_DIR}/trained_patch_kitti.png"
echo "Artifacts:  ${OUTPUT_DIR}/eval_000*_attacked_image.png and patch masks"
