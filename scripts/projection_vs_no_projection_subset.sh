#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Train/evaluate two otherwise identical small-subset runs:
#   1. No projection: standard 2D patch placement.
#   2. Projection: perspective projection onto the road plane.
#
# Default uses a pixel patch for speed and stability. Set
# PATCH_PARAMETRIZATION=diffusion to run the same comparison with diffusion.

export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-file:${PWD}/mlruns}"
PYTHON="${PYTHON:-python3}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-experiment_data/projection_vs_no_projection_subset_${RUN_ID}}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-projection_vs_no_projection_subset}"

SUBSET_SIZE="${SUBSET_SIZE:-8}"
EPOCHS="${EPOCHS:-1}"
INNER_STEPS="${INNER_STEPS:-3}"
PATCH_SIZE="${PATCH_SIZE:-300}"
PATCH_PARAMETRIZATION="${PATCH_PARAMETRIZATION:-pixel}"
PATCH_LR="${PATCH_LR:-0.1}"
OPTIMIZER="${OPTIMIZER:-adam}"

FLOW_W="${FLOW_W:-1.0}"
MDE_W="${MDE_W:-0.1}"
SS_W="${SS_W:-1.0}"
WEIGHT_STRATEGY="${WEIGHT_STRATEGY:-normalized}"
FLOW_TARGET_MAGNITUDE="${FLOW_TARGET_MAGNITUDE:-10.0}"
TV_WEIGHT="${TV_WEIGHT:-0.05}"
NPS_WEIGHT="${NPS_WEIGHT:-0.5}"

common_args=(
  --dataset Kitti15
  --eval_mode testing
  --model_name raft
  --mde_model depth-anything-v2
  --ss_model segformer_cityscapes
  --attack_mde
  --attack_ss
  --patch_parametrization "${PATCH_PARAMETRIZATION}"
  --patch_size "${PATCH_SIZE}"
  --n "${EPOCHS}"
  --steps "${INNER_STEPS}"
  --lr "${PATCH_LR}"
  --optimizer "${OPTIMIZER}"
  --loss_weights "${FLOW_W}" "${MDE_W}" "${SS_W}"
  --weight_strategy "${WEIGHT_STRATEGY}"
  --target down
  --down_loss hinge
  --mde_target near
  --ss_target targeted
  --flow_target_magnitude "${FLOW_TARGET_MAGNITUDE}"
  --tv_weight "${TV_WEIGHT}"
  --nps_weight "${NPS_WEIGHT}"
  --subset_size "${SUBSET_SIZE}"
  --save_patch_every 0
  --experiment_name "${EXPERIMENT_NAME}"
)

if [[ "${PATCH_PARAMETRIZATION}" == "diffusion" ]]; then
  common_args+=(
    --diffusion_model "${DIFFUSION_MODEL:-sdxl}"
    --diffusion_init_mode "${DIFFUSION_INIT_MODE:-image}"
    --diffusion_base_image "${DIFFUSION_BASE_IMAGE:-test_assets/dog.jpg}"
    --diffusion_prompt "${DIFFUSION_PROMPT:-dog logo}"
    --diffusion_decode_mode "${DIFFUSION_DECODE_MODE:-denoise}"
    --diffusion_source_steps "${DIFFUSION_SOURCE_STEPS:-50}"
    --diffusion_reverse_steps "${DIFFUSION_REVERSE_STEPS:-25}"
    --diffusion_guidance_scale "${DIFFUSION_GUIDANCE_SCALE:-3.0}"
    --diffusion_latent_eps "${DIFFUSION_LATENT_EPS:-1.0}"
    --diffusion_null_inner_steps "${DIFFUSION_NULL_INNER_STEPS:-15}"
  )
fi

mkdir -p "${OUTPUT_ROOT}"

for mode in no_projection projection; do
  output_dir="${OUTPUT_ROOT}/${mode}"
  args=("${common_args[@]}" --output_dir "${output_dir}")
  if [[ "${mode}" == "projection" ]]; then
    args+=(--patch_projection)
  fi

  echo "============================================================"
  echo "Running ${mode}: ${output_dir}"
  echo "============================================================"
  "${PYTHON}" run_patch_attack.py "${args[@]}" 2>&1 | tee "${output_dir}.log"
done

echo "Done. Results are in ${OUTPUT_ROOT}; MLflow experiment: ${EXPERIMENT_NAME}"
