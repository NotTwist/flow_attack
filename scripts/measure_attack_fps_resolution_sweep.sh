#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Measure end-to-end attacked inference throughput for the 3-model stack:
# patch overlay + optical flow + monocular depth + semantic segmentation.
#
# Resolution format is HEIGHTxWIDTH. Defaults include a KITTI-like endpoint.

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-codex}}"
mkdir -p "${MPLCONFIGDIR}"

PYTHON="${PYTHON:-python3}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-experiment_data/attack_fps_resolution_sweep_${RUN_ID}}"

cmd=(
  "${PYTHON}" scripts/measure_attack_fps_resolution_sweep.py
  --dataset "${DATASET:-Kitti15}"
  --eval_mode "${EVAL_MODE:-testing}"
  --model_name "${FLOW_MODEL:-raft}"
  --mde_model "${MDE_MODEL:-depth-anything-v2}"
  --ss_model "${SS_MODEL:-segformer_cityscapes}"
  --resolutions "${RESOLUTIONS:-192x640,256x832,320x1024,375x1242}"
  --num_batches "${NUM_BATCHES:-10}"
  --warmup "${WARMUP:-2}"
  --repeats "${REPEATS:-5}"
  --patch_size "${PATCH_SIZE:-100}"
  --flow_shift "${FLOW_SHIFT:-0}"
  --y_scale "${Y_SCALE:-3}"
  --output_dir "${OUTPUT_DIR}"
)

if [[ -n "${TRAINED_PATCH:-}" ]]; then
  cmd+=(--trained_patch "${TRAINED_PATCH}")
fi

"${cmd[@]}" "$@"

echo "Done. Results are in ${OUTPUT_DIR}"
