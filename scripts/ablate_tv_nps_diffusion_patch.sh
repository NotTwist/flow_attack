#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Diffusion patch TV/NPS ablation on KITTI15.
# Runs several regularization settings with the same task setup.
#
# Override ABLATION_GRID to change variants:
#   ABLATION_GRID="none:0:0 tv_only:0.05:0 nps_only:0:0.5 both:0.05:0.5"

export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-file:${PWD}/mlruns}"
PYTHON="${PYTHON:-python3}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-experiment_data/tv_nps_ablation_diffusion_kitti15_${RUN_ID}}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-tv_nps_ablation_diffusion_kitti15}"

SUBSET_SIZE="${SUBSET_SIZE:-8}"
EPOCHS="${EPOCHS:-1}"
INNER_STEPS="${INNER_STEPS:-3}"
PATCH_SIZE="${PATCH_SIZE:-300}"
PATCH_LR="${PATCH_LR:-0.03}"
ABLATION_GRID="${ABLATION_GRID:-none:0:0 tv_only:0.05:0 nps_only:0:0.5 both:0.05:0.5 strong_both:0.2:1.0}"

mkdir -p "${OUTPUT_ROOT}"

for spec in ${ABLATION_GRID}; do
  IFS=: read -r label tv_weight nps_weight <<< "${spec}"
  output_dir="${OUTPUT_ROOT}/${label}"
  echo "============================================================"
  echo "Running diffusion TV/NPS ablation: ${label}"
  echo "TV=${tv_weight}, NPS=${nps_weight}"
  echo "Output: ${output_dir}"
  echo "============================================================"
  "${PYTHON}" run_patch_attack.py \
    --dataset Kitti15 \
    --eval_mode testing \
    --model_name raft \
    --mde_model depth-anything-v2 \
    --ss_model segformer_cityscapes \
    --attack_mde \
    --attack_ss \
    --patch_parametrization diffusion \
    --patch_size "${PATCH_SIZE}" \
    --n "${EPOCHS}" \
    --steps "${INNER_STEPS}" \
    --lr "${PATCH_LR}" \
    --diffusion_optimizer "${DIFFUSION_OPTIMIZER:-adam}" \
    --loss_weights "${FLOW_W:-1.0}" "${MDE_W:-0.1}" "${SS_W:-1.0}" \
    --weight_strategy "${WEIGHT_STRATEGY:-normalized}" \
    --target down \
    --down_loss hinge \
    --mde_target near \
    --ss_target targeted \
    --flow_target_magnitude "${FLOW_TARGET_MAGNITUDE:-10.0}" \
    --tv_weight "${tv_weight}" \
    --nps_weight "${nps_weight}" \
    --diffusion_model "${DIFFUSION_MODEL:-sdxl}" \
    --diffusion_init_mode "${DIFFUSION_INIT_MODE:-image}" \
    --diffusion_base_image "${DIFFUSION_BASE_IMAGE:-test_assets/dog.jpg}" \
    --diffusion_prompt "${DIFFUSION_PROMPT:-dog logo}" \
    --diffusion_decode_mode "${DIFFUSION_DECODE_MODE:-denoise}" \
    --diffusion_source_steps "${DIFFUSION_SOURCE_STEPS:-50}" \
    --diffusion_reverse_steps "${DIFFUSION_REVERSE_STEPS:-25}" \
    --diffusion_guidance_scale "${DIFFUSION_GUIDANCE_SCALE:-3.0}" \
    --diffusion_latent_eps "${DIFFUSION_LATENT_EPS:-1.0}" \
    --diffusion_null_inner_steps "${DIFFUSION_NULL_INNER_STEPS:-15}" \
    --subset_size "${SUBSET_SIZE}" \
    --output_dir "${output_dir}" \
    --experiment_name "${EXPERIMENT_NAME}" \
    2>&1 | tee "${output_dir}.log"
done

echo "Done. Compare runs in MLflow experiment: ${EXPERIMENT_NAME}"
