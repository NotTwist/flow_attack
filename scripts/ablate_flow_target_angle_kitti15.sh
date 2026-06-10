#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Sweep the absolute target-flow direction used by the patch attack.
#
# Angle convention: image coordinates, 0=right, 90=down, 180=left, 270=up.
# Each angle trains and evaluates an independent patch with identical settings.
#
# Useful overrides:
#   ANGLES="45 90 135" SUBSET_SIZE=20 EPOCHS=3 bash scripts/ablate_flow_target_angle_kitti15.sh
#   PATCH_PARAMETRIZATION=diffusion bash scripts/ablate_flow_target_angle_kitti15.sh

export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-file:${PWD}/mlruns}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-codex}}"
mkdir -p "${MPLCONFIGDIR}"
PYTHON="${PYTHON:-python3}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-experiment_data/flow_target_angle_ablation_kitti15_${RUN_ID}}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-flow_target_angle_ablation_kitti15}"

ANGLES="${ANGLES:-0 45 90 135 180 225 270 315}"
SUBSET_SIZE="${SUBSET_SIZE:-12}"
EPOCHS="${EPOCHS:-2}"
INNER_STEPS="${INNER_STEPS:-5}"
PATCH_SIZE="${PATCH_SIZE:-300}"
PATCH_PARAMETRIZATION="${PATCH_PARAMETRIZATION:-pixel}"
PATCH_LR="${PATCH_LR:-0.1}"
FLOW_TARGET_MAGNITUDE="${FLOW_TARGET_MAGNITUDE:-10.0}"
DOWN_LOSS="${DOWN_LOSS:-hinge}"

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
  --target direction
  --mde_target near
  --ss_target targeted
  --flow_target_magnitude "${FLOW_TARGET_MAGNITUDE}"
  --down_loss "${DOWN_LOSS}"
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

for angle in ${ANGLES}; do
  angle_label="${angle//./p}"
  output_dir="${OUTPUT_ROOT}/angle_${angle_label}"
  echo "============================================================"
  echo "Running flow-target angle ablation: ${angle} deg"
  echo "Output: ${output_dir}"
  echo "============================================================"
  "${PYTHON}" run_patch_attack.py \
    "${common_args[@]}" \
    --flow_target_angle_deg "${angle}" \
    --output_dir "${output_dir}" \
    2>&1 | tee "${output_dir}.log"
done

"${PYTHON}" scripts/summarize_flow_target_angle_sweep.py "${OUTPUT_ROOT}"

echo "Done. Compare runs in MLflow experiment: ${EXPERIMENT_NAME}"
echo "Summary: ${OUTPUT_ROOT}/flow_target_angle_summary.csv"
