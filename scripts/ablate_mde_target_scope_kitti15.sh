#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Ablate MDE target construction:
#   near / far             : target is recomputed per image from that image's raw depth min/max
#   near_global / far_global: target is fixed from dataset-level raw depth min/max

export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-file:${PWD}/mlruns}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-codex}}"
mkdir -p "${MPLCONFIGDIR}"
PYTHON="${PYTHON:-python3}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-experiment_data/mde_target_scope_ablation_kitti15_${RUN_ID}}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-mde_target_scope_ablation_kitti15}"

MDE_TARGETS="${MDE_TARGETS:-near near_global far far_global}"
SUBSET_SIZE="${SUBSET_SIZE:-12}"
EPOCHS="${EPOCHS:-2}"
INNER_STEPS="${INNER_STEPS:-5}"
PATCH_SIZE="${PATCH_SIZE:-300}"
PATCH_PARAMETRIZATION="${PATCH_PARAMETRIZATION:-pixel}"
PATCH_LR="${PATCH_LR:-0.1}"
FLOW_TARGET="${FLOW_TARGET:-direction}"
FLOW_TARGET_ANGLE_DEG="${FLOW_TARGET_ANGLE_DEG:-180}"
FLOW_TARGET_MAGNITUDE="${FLOW_TARGET_MAGNITUDE:-10.0}"
MDE_NEAR_MARGIN="${MDE_NEAR_MARGIN:-0.1}"

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
  --target "${FLOW_TARGET}"
  --flow_target_angle_deg "${FLOW_TARGET_ANGLE_DEG}"
  --mde_near_margin "${MDE_NEAR_MARGIN}"
  --ss_target targeted
  --flow_target_magnitude "${FLOW_TARGET_MAGNITUDE}"
  --down_loss "${DOWN_LOSS:-hinge}"
  --down_hinge_min_mag_ratio "${DOWN_HINGE_MIN_MAG_RATIO:-0.8}"
  --down_hinge_horizontal_weight "${DOWN_HINGE_HORIZONTAL_WEIGHT:-0.1}"
  --down_hinge_magnitude_weight "${DOWN_HINGE_MAGNITUDE_WEIGHT:-0.5}"
  --down_hinge_vertical_weight "${DOWN_HINGE_VERTICAL_WEIGHT:-1.0}"
  --tv_weight "${TV_WEIGHT:-0.05}"
  --nps_weight "${NPS_WEIGHT:-0.5}"
  --subset_size "${SUBSET_SIZE}"
  --experiment_name "${EXPERIMENT_NAME}"
)

mkdir -p "${OUTPUT_ROOT}"

for mde_target in ${MDE_TARGETS}; do
  output_dir="${OUTPUT_ROOT}/${mde_target}"
  echo "============================================================"
  echo "Running MDE target scope ablation: ${mde_target}"
  echo "Output: ${output_dir}"
  echo "============================================================"
  "${PYTHON}" run_patch_attack.py \
    "${common_args[@]}" \
    --mde_target "${mde_target}" \
    --output_dir "${output_dir}" \
    2>&1 | tee "${output_dir}.log"
done

"${PYTHON}" scripts/summarize_mde_target_scope_ablation.py "${OUTPUT_ROOT}"

echo "Done. Compare runs in MLflow experiment: ${EXPERIMENT_NAME}"
echo "Summary: ${OUTPUT_ROOT}/mde_target_scope_summary.csv"
