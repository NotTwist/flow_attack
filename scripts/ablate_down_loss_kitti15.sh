#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Ablate standard EPE-to-down-target vs the new down hinge loss.
# Default is a small pixel-patch run, enough to check whether the large-patch
# zero-flow collapse returns under EPE.

export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-file:${PWD}/mlruns}"
PYTHON="${PYTHON:-python3}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-experiment_data/down_loss_ablation_kitti15_${RUN_ID}}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-down_loss_ablation_kitti15}"

SUBSET_SIZE="${SUBSET_SIZE:-12}"
EPOCHS="${EPOCHS:-2}"
INNER_STEPS="${INNER_STEPS:-5}"
PATCH_SIZE="${PATCH_SIZE:-300}"
PATCH_PARAMETRIZATION="${PATCH_PARAMETRIZATION:-pixel}"
PATCH_LR="${PATCH_LR:-0.1}"
FLOW_TARGET_MAGNITUDE="${FLOW_TARGET_MAGNITUDE:-10.0}"

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
  --optimizer adam
  --loss_weights "${FLOW_W:-1.0}" "${MDE_W:-0.1}" "${SS_W:-1.0}"
  --weight_strategy "${WEIGHT_STRATEGY:-normalized}"
  --target down
  --mde_target near
  --ss_target targeted
  --flow_target_magnitude "${FLOW_TARGET_MAGNITUDE}"
  --tv_weight "${TV_WEIGHT:-0.05}"
  --nps_weight "${NPS_WEIGHT:-0.5}"
  --subset_size "${SUBSET_SIZE}"
  --experiment_name "${EXPERIMENT_NAME}"
)

mkdir -p "${OUTPUT_ROOT}"

for down_loss in epe hinge; do
  output_dir="${OUTPUT_ROOT}/${down_loss}"
  echo "============================================================"
  echo "Running down loss ablation: ${down_loss}"
  echo "Output: ${output_dir}"
  echo "============================================================"
  "${PYTHON}" run_patch_attack.py \
    "${common_args[@]}" \
    --down_loss "${down_loss}" \
    --output_dir "${output_dir}" \
    2>&1 | tee "${output_dir}.log"
done

echo "Done. Compare runs in MLflow experiment: ${EXPERIMENT_NAME}"
