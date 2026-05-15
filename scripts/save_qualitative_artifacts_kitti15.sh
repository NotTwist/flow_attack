#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Save qualitative before/after artifacts for the latest diffusion and pixel
# patch checkpoints. Uses a tiny subset by default so it creates 2-3 frames.
#
# Outputs:
#   experiment_data/qualitative_artifacts_kitti15/<kind>/diploma_artifacts/

export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-file:${PWD}/mlruns}"
PYTHON="${PYTHON:-python3}"

OUTPUT_ROOT="${OUTPUT_ROOT:-experiment_data/qualitative_artifacts_kitti15}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qualitative_artifacts_kitti15}"
SUBSET_SIZE="${SUBSET_SIZE:-3}"
EVAL_MODE="${EVAL_MODE:-testing}"
PATCH_SIZE="${PATCH_SIZE:-300}"
EVAL_ARTIFACT_LIMIT="${EVAL_ARTIFACT_LIMIT:-3}"
ALLOW_PARTIAL="${ALLOW_PARTIAL:-false}"

allow_partial_args=()
if [[ "${ALLOW_PARTIAL}" == "true" ]]; then
  allow_partial_args+=(--allow_partial)
fi

mkdir -p "${OUTPUT_ROOT}"

for kind in diffusion pixel; do
  metadata="$("${PYTHON}" scripts/find_latest_patch_checkpoint.py --kind "${kind}" "${allow_partial_args[@]}")"
  echo "============================================================"
  echo "Saving qualitative artifacts for ${kind}"
  echo "Metadata: ${metadata}"
  echo "============================================================"
  "${PYTHON}" scripts/eval_patch_from_metadata.py \
    --metadata "${metadata}" \
    --output_dir "${OUTPUT_ROOT}/${kind}" \
    --experiment_name "${EXPERIMENT_NAME}" \
    --eval_mode "${EVAL_MODE}" \
    --subset_size "${SUBSET_SIZE}" \
    --save_artifacts \
    --eval_artifact_limit "${EVAL_ARTIFACT_LIMIT}" \
    -- \
    --save_diploma_artifacts \
    --patch_size "${PATCH_SIZE}"
done

echo "Artifacts saved under ${OUTPUT_ROOT}"
