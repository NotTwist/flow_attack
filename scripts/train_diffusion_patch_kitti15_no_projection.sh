#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Full diffusion-patch training on KITTI15 without surface projection.
#
# This is intended as a normal experiment, not a smoke test:
# - no --small_run
# - no --subset_size unless explicitly provided
# - no --patch_projection
# - diffusion patch initialized from a manhole image
# - flow + MDE + semantic segmentation objectives enabled by default
#
# Example:
#   bash scripts/train_diffusion_patch_kitti15_no_projection.sh
#
# Faster but still non-smoke:
#   EPOCHS=10 INNER_STEPS=5 bash scripts/train_diffusion_patch_kitti15_no_projection.sh
#
# Full denoising each patch render:
#   DIFFUSION_REVERSE_STEPS=50 bash scripts/train_diffusion_patch_kitti15_no_projection.sh

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-experiment_data/diffusion_patch_kitti15_no_projection_${RUN_ID}}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-diffusion_patch_kitti15_no_projection}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-file:${PWD}/mlruns}"

DATASET="${DATASET:-Kitti15}"
EVAL_MODE="${EVAL_MODE:-testing}"
FLOW_MODEL="${FLOW_MODEL:-raft}"
MDE_MODEL="${MDE_MODEL:-depth-anything-v2}"
SS_MODEL="${SS_MODEL:-segformer_cityscapes}"

EPOCHS="${EPOCHS:-10}"
INNER_STEPS="${INNER_STEPS:-5}"
PATCH_SIZE="${PATCH_SIZE:-300}"
PATCH_LR="${PATCH_LR:-0.03}"
MAX_DELTA="${MAX_DELTA:-0.008}"
DIFFUSION_OPTIMIZER="${DIFFUSION_OPTIMIZER:-adam}"
SAVE_PATCH_EVERY="${SAVE_PATCH_EVERY:-10}"
PATCH_CHECKPOINT_DIR="${PATCH_CHECKPOINT_DIR:-${OUTPUT_DIR}/patch_checkpoints}"

FLOW_W="${FLOW_W:-1.0}"
MDE_W="${MDE_W:-0.1}"
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

TV_WEIGHT="${TV_WEIGHT:-0.05}"
NPS_WEIGHT="${NPS_WEIGHT:-0.5}"
Y_SCALE="${Y_SCALE:-1}"
FLOW_SHIFT="${FLOW_SHIFT:-0}"

DIFFUSION_MODEL="${DIFFUSION_MODEL:-sdxl}"
DIFFUSION_INIT_MODE="${DIFFUSION_INIT_MODE:-image}"
DIFFUSION_BASE_IMAGE="${DIFFUSION_BASE_IMAGE:-test_assets/dog.jpg}"
DIFFUSION_PROMPT="${DIFFUSION_PROMPT:-dog logo}"
DIFFUSION_DTYPE="${DIFFUSION_DTYPE:-auto}"
DIFFUSION_DECODE_MODE="${DIFFUSION_DECODE_MODE:-denoise}"
DIFFUSION_SOURCE_STEPS="${DIFFUSION_SOURCE_STEPS:-50}"
DIFFUSION_REVERSE_STEPS="${DIFFUSION_REVERSE_STEPS:-25}"
DIFFUSION_GUIDANCE_SCALE="${DIFFUSION_GUIDANCE_SCALE:-3.0}"
DIFFUSION_LATENT_EPS="${DIFFUSION_LATENT_EPS:-1}"
DIFFUSION_NULL_INNER_STEPS="${DIFFUSION_NULL_INNER_STEPS:-15}"
DIFFUSION_NULL_EPSILON="${DIFFUSION_NULL_EPSILON:-1e-5}"
DIFFUSION_SEED="${DIFFUSION_SEED:-42}"

SUBSET_SIZE="${SUBSET_SIZE:-0}"

subset_args=()
if [[ "${SUBSET_SIZE}" != "0" ]]; then
  subset_args+=(--subset_size "${SUBSET_SIZE}")
fi

mkdir -p "${OUTPUT_DIR}/logs"

echo "============================================================"
echo "Diffusion patch training on KITTI15 without projection"
echo "Run id:          ${RUN_ID}"
echo "Output dir:      ${OUTPUT_DIR}"
echo "MLflow URI:      ${MLFLOW_TRACKING_URI}"
echo "Experiment:      ${EXPERIMENT_NAME}"
echo "Dataset/eval:    ${DATASET} / ${EVAL_MODE}"
echo "Models:          flow=${FLOW_MODEL}, mde=${MDE_MODEL}, ss=${SS_MODEL}"
echo "Training:        epochs=${EPOCHS}, inner_steps=${INNER_STEPS}, lr=${PATCH_LR}"
echo "Patch:           size=${PATCH_SIZE}, diffusion=${DIFFUSION_MODEL}, reverse_steps=${DIFFUSION_REVERSE_STEPS}"
echo "Decode mode:     ${DIFFUSION_DECODE_MODE}"
echo "Physical regs:   tv=${TV_WEIGHT}, nps=${NPS_WEIGHT}"
echo "Down hinge:      min_mag_ratio=${DOWN_HINGE_MIN_MAG_RATIO}, horiz=${DOWN_HINGE_HORIZONTAL_WEIGHT}, mag=${DOWN_HINGE_MAGNITUDE_WEIGHT}, vert=${DOWN_HINGE_VERTICAL_WEIGHT}"
echo "Down loss:       ${DOWN_LOSS}"
echo "Latent eps:      ${DIFFUSION_LATENT_EPS}"
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
  --patch_parametrization diffusion \
  --patch_size "${PATCH_SIZE}" \
  --n "${EPOCHS}" \
  --steps "${INNER_STEPS}" \
  --lr "${PATCH_LR}" \
  --max_delta "${MAX_DELTA}" \
  --diffusion_optimizer "${DIFFUSION_OPTIMIZER}" \
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
  --diffusion_model "${DIFFUSION_MODEL}" \
  --diffusion_init_mode "${DIFFUSION_INIT_MODE}" \
  --diffusion_base_image "${DIFFUSION_BASE_IMAGE}" \
  --diffusion_prompt "${DIFFUSION_PROMPT}" \
  --diffusion_dtype "${DIFFUSION_DTYPE}" \
  --diffusion_decode_mode "${DIFFUSION_DECODE_MODE}" \
  --diffusion_source_steps "${DIFFUSION_SOURCE_STEPS}" \
  --diffusion_reverse_steps "${DIFFUSION_REVERSE_STEPS}" \
  --diffusion_guidance_scale "${DIFFUSION_GUIDANCE_SCALE}" \
  --diffusion_latent_eps "${DIFFUSION_LATENT_EPS}" \
  --diffusion_null_inner_steps "${DIFFUSION_NULL_INNER_STEPS}" \
  --diffusion_null_epsilon "${DIFFUSION_NULL_EPSILON}" \
  --diffusion_seed "${DIFFUSION_SEED}" \
  --output_dir "${OUTPUT_DIR}" \
  --experiment_name "${EXPERIMENT_NAME}" \
  "${subset_args[@]}" \
  2>&1 | tee "${OUTPUT_DIR}/logs/train.log"

echo "============================================================"
echo "Training finished."
echo "Output dir: ${OUTPUT_DIR}"
echo "MLflow:     mlflow ui --backend-store-uri ${MLFLOW_TRACKING_URI#file:}"
echo "============================================================"
