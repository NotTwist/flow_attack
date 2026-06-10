#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Sweep target-flow magnitudes and directions used by the patch attack.
#
# Angle convention follows image coordinates: 0=right, 90=down,
# 180=left, 270=up. Each (angle, magnitude) pair trains/evaluates an
# independent patch with identical non-target settings.
#
# Useful overrides:
#   ANGLES="135 180 225 270" MAGNITUDES="0.5 1 2 5 10 20 40" bash scripts/ablate_flow_target_magnitude_kitti15.sh
#   PATCH_PARAMETRIZATION=diffusion EPOCHS=3 bash scripts/ablate_flow_target_magnitude_kitti15.sh

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-codex}}"
mkdir -p "${MPLCONFIGDIR}"
PYTHON="${PYTHON:-python3}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-experiment_data/flow_target_magnitude_ablation_kitti15_${RUN_ID}}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-flow_target_magnitude_ablation_kitti15}"

if [[ "${KEEP_MLFLOW:-0}" == "1" ]]; then
  export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-file:${PWD}/mlruns}"
else
  export MLFLOW_TRACKING_URI="file:${PWD}/${OUTPUT_ROOT}/mlruns"
fi

MAGNITUDES="${MAGNITUDES:-0.25 0.5 1 2 5 10 20 40 80}"
if [[ -n "${FLOW_TARGET_ANGLE_DEG:-}" && -z "${ANGLES:-}" ]]; then
  ANGLES="${FLOW_TARGET_ANGLE_DEG}"
else
  ANGLES="${ANGLES:-0 45 90 135 180 225 270 315}"
fi
SUBSET_SIZE="${SUBSET_SIZE:-12}"
EPOCHS="${EPOCHS:-2}"
INNER_STEPS="${INNER_STEPS:-5}"
PATCH_SIZE="${PATCH_SIZE:-300}"
PATCH_PARAMETRIZATION="${PATCH_PARAMETRIZATION:-pixel}"
PATCH_LR="${PATCH_LR:-0.1}"

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
  --mde_target "${MDE_TARGET:-near}"
  --mde_near_margin "${MDE_NEAR_MARGIN:-0.1}"
  --ss_target targeted
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

for angle in ${ANGLES}; do
  angle_label="${angle//./p}"
  for magnitude in ${MAGNITUDES}; do
    magnitude_label="${magnitude//./p}"
    output_dir="${OUTPUT_ROOT}/angle_${angle_label}_mag_${magnitude_label}"
    log_path="${output_dir}.log"
    if [[ "${SKIP_COMPLETED:-1}" == "1" ]] && [[ -f "${log_path}" ]] && grep -q "Final AEE Metrics:" "${log_path}"; then
      echo "Skipping completed run: angle=${angle}, magnitude=${magnitude}"
      continue
    fi
    echo "============================================================"
    echo "Running flow-target magnitude/direction ablation"
    echo "Direction angle: ${angle} deg"
    echo "Target magnitude: ${magnitude}"
    echo "Output: ${output_dir}"
    echo "============================================================"
    set +e
    "${PYTHON}" run_patch_attack.py \
      "${common_args[@]}" \
      --flow_target_angle_deg "${angle}" \
      --flow_target_magnitude "${magnitude}" \
      --output_dir "${output_dir}" \
      2>&1 | tee "${log_path}"
    command_status=("${PIPESTATUS[@]}")
    set -e
    if [[ "${REMOVE_PATCH_CHECKPOINTS:-1}" == "1" ]]; then
      rm -rf "${output_dir}/patch_checkpoints"
    fi
    if [[ "${KEEP_MLFLOW:-0}" != "1" && "${REMOVE_MLFLOW_AFTER_RUN:-1}" == "1" ]]; then
      rm -rf "${OUTPUT_ROOT}/mlruns"
    fi
    if [[ "${command_status[0]}" != "0" ]]; then
      echo "Run failed: angle=${angle}, magnitude=${magnitude}" >&2
      exit "${command_status[0]}"
    fi
    if [[ "${command_status[1]}" != "0" ]]; then
      echo "Log writing failed: ${log_path}" >&2
      exit "${command_status[1]}"
    fi
  done
done

"${PYTHON}" scripts/summarize_flow_target_magnitude_sweep.py "${OUTPUT_ROOT}"

echo "Done. Compare runs in MLflow experiment: ${EXPERIMENT_NAME}"
echo "Summary: ${OUTPUT_ROOT}/flow_target_magnitude_summary.csv"
