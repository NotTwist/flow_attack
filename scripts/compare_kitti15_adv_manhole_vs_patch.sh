#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Compare adv-manhole and this project's physical patch attack on the same
# KITTI15/model/evaluation setup. Override any value from the shell:
#
#   SMALL_RUN=1 EPOCHS=2 SUBSET_SIZE=16 bash scripts/compare_kitti15_adv_manhole_vs_patch.sh
#
# The two methods do not optimize exactly the same objective internally, but
# this script aligns the dataset, evaluation split, models, patch footprint,
# random seed path, and output/MLflow naming.

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-experiment_data/kitti15_equal_comparison_${RUN_ID}}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-kitti15_adv_manhole_vs_patch_${RUN_ID}}"

GPU="${GPU:-0}"
SMALL_RUN="${SMALL_RUN:-0}"
SUBSET_SIZE="${SUBSET_SIZE:-0}"
EVAL_MODE="${EVAL_MODE:-testing}"
SAVE_ARTIFACTS="${SAVE_ARTIFACTS:-0}"

FLOW_MODEL="${FLOW_MODEL:-raft}"
MDE_MODEL="${MDE_MODEL:-depth-anything-v2}"
SS_MODEL="${SS_MODEL:-segformer_cityscapes}"
ROAD_CLASS_IDX="${ROAD_CLASS_IDX:-0}"

EPOCHS="${EPOCHS:-25}"
BATCH_SIZE="${BATCH_SIZE:-1}"
PATCH_SIZE="${PATCH_SIZE:-100}"
TEX_SCALE="${TEX_SCALE:-100.0}"

# Objective knobs for this project's patch attack.
# Defaults emphasize MDE+SS because adv-manhole optimizes those tasks, while
# optical flow is evaluated as a transferred downstream effect.
FLOW_W="${FLOW_W:-0.0}"
MDE_W="${MDE_W:-2.0}"
SS_W="${SS_W:-1.0}"
PATCH_PARAMETRIZATION="${PATCH_PARAMETRIZATION:-pixel}"
PATCH_LR="${PATCH_LR:-0.1}"
PATCH_INNER_STEPS="${PATCH_INNER_STEPS:-20}"
FLOW_TARGET="${FLOW_TARGET:-zero}"
MDE_TARGET="${MDE_TARGET:-zero}"
SS_TARGET="${SS_TARGET:-targeted}"
FLOW_TARGET_MAGNITUDE="${FLOW_TARGET_MAGNITUDE:-1.0}"
Y_SCALE="${Y_SCALE:-1}"
FLOW_SHIFT="${FLOW_SHIFT:-0}"
TV_WEIGHT="${TV_WEIGHT:-0}"
NPS_WEIGHT="${NPS_WEIGHT:-0}"

# Diffusion-only knobs. These are used only when PATCH_PARAMETRIZATION=diffusion.
DIFFUSION_MODEL="${DIFFUSION_MODEL:-sdxl}"
DIFFUSION_PROMPT="${DIFFUSION_PROMPT:-a circular road manhole cover}"
DIFFUSION_INIT_MODE="${DIFFUSION_INIT_MODE:-random}"
DIFFUSION_BASE_IMAGE="${DIFFUSION_BASE_IMAGE:-}"
DIFFUSION_SOURCE_STEPS="${DIFFUSION_SOURCE_STEPS:-50}"
DIFFUSION_REVERSE_STEPS="${DIFFUSION_REVERSE_STEPS:-25}"
DIFFUSION_GUIDANCE_SCALE="${DIFFUSION_GUIDANCE_SCALE:-7.5}"
DIFFUSION_LATENT_EPS="${DIFFUSION_LATENT_EPS:-0.5}"
DIFFUSION_DTYPE="${DIFFUSION_DTYPE:-auto}"

# Objective knobs for adv-manhole.
ADV_LR="${ADV_LR:-0.01}"
ADV_TRAIN_FRACTION="${ADV_TRAIN_FRACTION:-0.8}"

mkdir -p "${OUTPUT_ROOT}/logs"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-file:${PWD}/mlruns}"

common_subset_args=()
if [[ "${SMALL_RUN}" == "1" ]]; then
  common_subset_args+=(--small_run)
fi
if [[ "${SUBSET_SIZE}" != "0" ]]; then
  common_subset_args+=(--subset_size "${SUBSET_SIZE}")
fi

artifact_args=()
if [[ "${SAVE_ARTIFACTS}" == "1" ]]; then
  artifact_args+=(--save_artifacts)
fi

diffusion_args=()
if [[ "${PATCH_PARAMETRIZATION}" == "diffusion" ]]; then
  diffusion_args+=(
    --diffusion_model "${DIFFUSION_MODEL}"
    --diffusion_prompt "${DIFFUSION_PROMPT}"
    --diffusion_init_mode "${DIFFUSION_INIT_MODE}"
    --diffusion_source_steps "${DIFFUSION_SOURCE_STEPS}"
    --diffusion_reverse_steps "${DIFFUSION_REVERSE_STEPS}"
    --diffusion_guidance_scale "${DIFFUSION_GUIDANCE_SCALE}"
    --diffusion_latent_eps "${DIFFUSION_LATENT_EPS}"
    --diffusion_dtype "${DIFFUSION_DTYPE}"
  )
  if [[ -n "${DIFFUSION_BASE_IMAGE}" ]]; then
    diffusion_args+=(--diffusion_base_image "${DIFFUSION_BASE_IMAGE}")
  fi
fi

echo "============================================================"
echo "KITTI15 equal-environment comparison"
echo "Run id:          ${RUN_ID}"
echo "Output root:     ${OUTPUT_ROOT}"
echo "MLflow URI:      ${MLFLOW_TRACKING_URI}"
echo "Experiment name: ${EXPERIMENT_NAME}"
echo "Eval split:      ${EVAL_MODE}"
echo "Models:          flow=${FLOW_MODEL}, mde=${MDE_MODEL}, ss=${SS_MODEL}"
echo "Subset:          small_run=${SMALL_RUN}, subset_size=${SUBSET_SIZE}"
echo "============================================================"

echo
echo ">>> Running this project's patch attack"
python3 run_patch_attack.py \
  --dataset Kitti15 \
  --model_name "${FLOW_MODEL}" \
  --mde_model "${MDE_MODEL}" \
  --ss_model "${SS_MODEL}" \
  --attack_mde \
  --attack_ss \
  --patch_projection \
  --patch_parametrization "${PATCH_PARAMETRIZATION}" \
  --patch_size "${PATCH_SIZE}" \
  --n "${EPOCHS}" \
  --steps "${PATCH_INNER_STEPS}" \
  --lr "${PATCH_LR}" \
  --loss_weights "${FLOW_W}" "${MDE_W}" "${SS_W}" \
  --target "${FLOW_TARGET}" \
  --mde_target "${MDE_TARGET}" \
  --ss_target "${SS_TARGET}" \
  --flow_target_magnitude "${FLOW_TARGET_MAGNITUDE}" \
  --y_scale "${Y_SCALE}" \
  --flow_shift "${FLOW_SHIFT}" \
  --tv_weight "${TV_WEIGHT}" \
  --nps_weight "${NPS_WEIGHT}" \
  --output_dir "${OUTPUT_ROOT}/our_patch" \
  --experiment_name "${EXPERIMENT_NAME}" \
  --eval_mode "${EVAL_MODE}" \
  "${common_subset_args[@]}" \
  "${artifact_args[@]}" \
  "${diffusion_args[@]}" \
  2>&1 | tee "${OUTPUT_ROOT}/logs/our_patch.log"

echo
echo ">>> Running adv-manhole KITTI15 adaptation"
python3 run_adv_manhole_kitti.py \
  --flow_model "${FLOW_MODEL}" \
  --mde_model "${MDE_MODEL}" \
  --ss_model "${SS_MODEL}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${ADV_LR}" \
  --train_fraction "${ADV_TRAIN_FRACTION}" \
  --tex_scale "${TEX_SCALE}" \
  --output_dir "${OUTPUT_ROOT}/adv_manhole" \
  --experiment_name "${EXPERIMENT_NAME}" \
  --eval_mode "${EVAL_MODE}" \
  --gpu "${GPU}" \
  --road_class_idx "${ROAD_CLASS_IDX}" \
  "${common_subset_args[@]}" \
  2>&1 | tee "${OUTPUT_ROOT}/logs/adv_manhole.log"

echo
echo "============================================================"
echo "Comparison finished."
echo "Outputs: ${OUTPUT_ROOT}"
echo "MLflow:  mlflow ui --backend-store-uri ${MLFLOW_TRACKING_URI#file:}"
echo "============================================================"
