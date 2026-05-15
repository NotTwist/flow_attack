#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

SMALL_RUN="${SMALL_RUN:-1}" \
SUBSET_SIZE="${SUBSET_SIZE:-8}" \
EPOCHS="${EPOCHS:-1}" \
PATCH_INNER_STEPS="${PATCH_INNER_STEPS:-2}" \
SAVE_ARTIFACTS="${SAVE_ARTIFACTS:-0}" \
bash scripts/compare_kitti15_adv_manhole_vs_patch.sh
