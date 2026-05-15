#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Full pixel-patch training on KITTI15 without surface projection.
# Uses the same task setup as train_diffusion_patch_kitti15_no_projection.sh,
# but optimizes RGB patch pixels directly instead of a diffusion latent.

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-experiment_data/pixel_patch_kitti15_no_projection_${RUN_ID}}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-pixel_patch_kitti15_no_projection}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-file:${PWD}/mlruns}"

DATASET="${DATASET:-Kitti15}"
EVAL_MODE="${EVAL_MODE:-testing}"
FLOW_MODEL="${FLOW_MODEL:-raft}"
MDE_MODEL="${MDE_MODEL:-depth-anything-v2}"
SS_MODEL="${SS_MODEL:-segformer_cityscapes}"

EPOCHS="${EPOCHS:-10}"
INNER_STEPS="${INNER_STEPS:-5}"
PATCH_SIZE="${PATCH_SIZE:-300}"
PATCH_LR="${PATCH_LR:-0.1}"
PIXEL_OPTIMIZER="${PIXEL_OPTIMIZER:-adam}"
CHANGE_OF_VARIABLES="${CHANGE_OF_VARIABLES:-true}"
MAX_DELTA="${MAX_DELTA:-0.008}"
SAVE_PATCH_EVERY="${SAVE_PATCH_EVERY:-10}"
PATCH_CHECKPOINT_DIR="${PATCH_CHECKPOINT_DIR:-${OUTPUT_DIR}/patch_checkpoints}"

FLOW_W="${FLOW_W:-0}"
MDE_W="${MDE_W:-1}"
SS_W="${SS_W:-1.0}"
WEIGHT_STRATEGY="${WEIGHT_STRATEGY:-normalized}"
FLOW_TARGET="${FLOW_TARGET:-down}"
DOWN_LOSS="${DOWN_LOSS:-hinge}"
MDE_TARGET="${MDE_TARGET:-near}"
SS_TARGET="${SS_TARGET:-targeted}"
SS_FOCAL_GAMMA="${SS_FOCAL_GAMMA:-2.0}"
FLOW_TARGET_MAGNITUDE="${FLOW_TARGET_MAGNITUDE:-10.0}"
DOWN_HINGE_MIN_MAG_RATIO="${DOWN_HINGE_MIN_MAG_RATIO:-0.8}"
DOWN_HINGE_HORIZONTAL_WEIGHT="${DOWN_HINGE_HORIZONTAL_WEIGHT:-0.1}"
DOWN_HINGE_MAGNITUDE_WEIGHT="${DOWN_HINGE_MAGNITUDE_WEIGHT:-0.5}"
DOWN_HINGE_VERTICAL_WEIGHT="${DOWN_HINGE_VERTICAL_WEIGHT:-1.0}"
MDE_NEAR_MARGIN="${MDE_NEAR_MARGIN:-0.1}"

TV_WEIGHT="${TV_WEIGHT:-1}"
NPS_WEIGHT="${NPS_WEIGHT:-0.5}"
Y_SCALE="${Y_SCALE:-1}"
FLOW_SHIFT="${FLOW_SHIFT:-0}"

SUBSET_SIZE="${SUBSET_SIZE:-0}"

extra_args=()
if [[ "${SUBSET_SIZE}" != "0" ]]; then
  extra_args+=(--subset_size "${SUBSET_SIZE}")
fi
if [[ "${CHANGE_OF_VARIABLES}" == "true" ]]; then
  extra_args+=(--change_of_variables)
fi

mkdir -p "${OUTPUT_DIR}/logs"

echo "============================================================"
echo "Pixel patch training on KITTI15 without projection"
echo "Run id:          ${RUN_ID}"
echo "Output dir:      ${OUTPUT_DIR}"
echo "MLflow URI:      ${MLFLOW_TRACKING_URI}"
echo "Experiment:      ${EXPERIMENT_NAME}"
echo "Dataset/eval:    ${DATASET} / ${EVAL_MODE}"
echo "Models:          flow=${FLOW_MODEL}, mde=${MDE_MODEL}, ss=${SS_MODEL}"
echo "Training:        epochs=${EPOCHS}, inner_steps=${INNER_STEPS}, lr=${PATCH_LR}, optimizer=${PIXEL_OPTIMIZER}"
echo "Patch:           size=${PATCH_SIZE}, parametrization=pixel"
echo "Physical regs:   tv=${TV_WEIGHT}, nps=${NPS_WEIGHT}"
echo "Down hinge:      min_mag_ratio=${DOWN_HINGE_MIN_MAG_RATIO}, horiz=${DOWN_HINGE_HORIZONTAL_WEIGHT}, mag=${DOWN_HINGE_MAGNITUDE_WEIGHT}, vert=${DOWN_HINGE_VERTICAL_WEIGHT}"
echo "Down loss:       ${DOWN_LOSS}"
echo "Patch ckpts:     every ${SAVE_PATCH_EVERY} batches -> ${PATCH_CHECKPOINT_DIR}"
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
  --patch_parametrization pixel \
  --patch_size "${PATCH_SIZE}" \
  --n "${EPOCHS}" \
  --steps "${INNER_STEPS}" \
  --lr "${PATCH_LR}" \
  --optimizer "${PIXEL_OPTIMIZER}" \
  --max_delta "${MAX_DELTA}" \
  --save_patch_every "${SAVE_PATCH_EVERY}" \
  --patch_checkpoint_dir "${PATCH_CHECKPOINT_DIR}" \
  --loss_weights "${FLOW_W}" "${MDE_W}" "${SS_W}" \
  --weight_strategy "${WEIGHT_STRATEGY}" \
  --target "${FLOW_TARGET}" \
  --down_loss "${DOWN_LOSS}" \
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
  "${extra_args[@]}" \
  2>&1 | tee "${OUTPUT_DIR}/logs/train.log"

echo "============================================================"
echo "Training finished."
echo "Output dir: ${OUTPUT_DIR}"
echo "MLflow:     mlflow ui --backend-store-uri ${MLFLOW_TRACKING_URI#file:}"
echo "============================================================"
