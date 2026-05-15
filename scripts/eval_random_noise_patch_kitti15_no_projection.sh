#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Random-noise pixel patch baseline on KITTI15 without surface projection.
# This skips training and evaluates a fixed random RGB patch with the same
# models, targets, patch size, and evaluation split used by the trained patches.

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-experiment_data/random_noise_patch_kitti15_no_projection_${RUN_ID}}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-random_noise_patch_kitti15_no_projection}"
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
WEIGHT_STRATEGY="${WEIGHT_STRATEGY:-normalized}"
FLOW_TARGET="${FLOW_TARGET:-down}"
MDE_TARGET="${MDE_TARGET:-near}"
SS_TARGET="${SS_TARGET:-targeted}"
SS_FOCAL_GAMMA="${SS_FOCAL_GAMMA:-2.0}"
FLOW_TARGET_MAGNITUDE="${FLOW_TARGET_MAGNITUDE:-10.0}"
DOWN_HINGE_MIN_MAG_RATIO="${DOWN_HINGE_MIN_MAG_RATIO:-0.8}"
DOWN_HINGE_HORIZONTAL_WEIGHT="${DOWN_HINGE_HORIZONTAL_WEIGHT:-0.1}"
DOWN_HINGE_MAGNITUDE_WEIGHT="${DOWN_HINGE_MAGNITUDE_WEIGHT:-0.5}"
DOWN_HINGE_VERTICAL_WEIGHT="${DOWN_HINGE_VERTICAL_WEIGHT:-1.0}"
MDE_NEAR_MARGIN="${MDE_NEAR_MARGIN:-0.1}"

TV_WEIGHT="${TV_WEIGHT:-0.05}"
NPS_WEIGHT="${NPS_WEIGHT:-0.5}"
Y_SCALE="${Y_SCALE:-1}"
FLOW_SHIFT="${FLOW_SHIFT:-0}"

SUBSET_SIZE="${SUBSET_SIZE:-0}"

subset_args=()
if [[ "${SUBSET_SIZE}" != "0" ]]; then
  subset_args+=(--subset_size "${SUBSET_SIZE}")
fi

mkdir -p "${OUTPUT_DIR}/logs"

echo "============================================================"
echo "Random-noise pixel patch baseline on KITTI15 without projection"
echo "Run id:          ${RUN_ID}"
echo "Output dir:      ${OUTPUT_DIR}"
echo "MLflow URI:      ${MLFLOW_TRACKING_URI}"
echo "Experiment:      ${EXPERIMENT_NAME}"
echo "Dataset/eval:    ${DATASET} / ${EVAL_MODE}"
echo "Models:          flow=${FLOW_MODEL}, mde=${MDE_MODEL}, ss=${SS_MODEL}"
echo "Patch:           size=${PATCH_SIZE}, parametrization=pixel, baseline=random"
echo "Physical regs:   tv=${TV_WEIGHT}, nps=${NPS_WEIGHT}"
echo "Down hinge:      min_mag_ratio=${DOWN_HINGE_MIN_MAG_RATIO}, horiz=${DOWN_HINGE_HORIZONTAL_WEIGHT}, mag=${DOWN_HINGE_MAGNITUDE_WEIGHT}, vert=${DOWN_HINGE_VERTICAL_WEIGHT}"
echo "Projection:      disabled"
echo "============================================================"

python3 run_patch_attack.py \
  --dataset "${DATASET}" \
  --eval_mode "${EVAL_MODE}" \
  --model_name "${FLOW_MODEL}" \
  --mde_model "${MDE_MODEL}" \
  --ss_model "${SS_MODEL}" \
  --attack_mde \
  --attack_ss \
  --baseline \
  --patch_parametrization pixel \
  --patch_size "${PATCH_SIZE}" \
  --loss_weights "${FLOW_W}" "${MDE_W}" "${SS_W}" \
  --weight_strategy "${WEIGHT_STRATEGY}" \
  --target "${FLOW_TARGET}" \
  --mde_target "${MDE_TARGET}" \
  --mde_near_margin "${MDE_NEAR_MARGIN}" \
  --ss_target "${SS_TARGET}" \
  --ss_focal_gamma "${SS_FOCAL_GAMMA}" \
  --flow_target_magnitude "${FLOW_TARGET_MAGNITUDE}" \
  --down_hinge_min_mag_ratio "${DOWN_HINGE_MIN_MAG_RATIO}" \
  --down_hinge_horizontal_weight "${DOWN_HINGE_HORIZONTAL_WEIGHT}" \
  --down_hinge_magnitude_weight "${DOWN_HINGE_MAGNITUDE_WEIGHT}" \
  --down_hinge_vertical_weight "${DOWN_HINGE_VERTICAL_WEIGHT}" \
  --tv_weight "${TV_WEIGHT}" \
  --nps_weight "${NPS_WEIGHT}" \
  --y_scale "${Y_SCALE}" \
  --flow_shift "${FLOW_SHIFT}" \
  --output_dir "${OUTPUT_DIR}" \
  --experiment_name "${EXPERIMENT_NAME}" \
  "${subset_args[@]}" \
  2>&1 | tee "${OUTPUT_DIR}/logs/eval.log"

echo "============================================================"
echo "Baseline evaluation finished."
echo "Output dir: ${OUTPUT_DIR}"
echo "Baseline patch: ${OUTPUT_DIR}/baseline_patch.png"
echo "MLflow:     mlflow ui --backend-store-uri ${MLFLOW_TRACKING_URI#file:}"
echo "============================================================"
