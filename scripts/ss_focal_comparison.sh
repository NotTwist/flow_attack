#!/bin/bash
cd "$(dirname "$0")/.."

# Compare SS attack with focal loss (gamma=2) vs plain cross-entropy (gamma=0).
# Only SS is attacked (no OF/MDE) to isolate the effect.

COMMON="--attack_mde --small_run --attack_ss --patch_projection --patch_size 100 --y_scale 3 --n 5 --steps 20 --loss_weights 0 0 1.0 --experiment_name ss_focal_comparison"

# ── 1. Focal CE (gamma=2, current default) ──────────────────────────
echo "=== SS attack: Focal CE (gamma=2.0) ==="
python3 run_patch_attack.py $COMMON \
  --ss_focal_gamma 2.0

# ── 2. Focal CE (gamma=1, milder focal) ─────────────────────────────
echo ""
echo "=== SS attack: Focal CE (gamma=1.0) ==="
python3 run_patch_attack.py $COMMON \
  --ss_focal_gamma 1.0

# ── 3. Plain CE (gamma=0, no focal weighting) ───────────────────────
echo ""
echo "=== SS attack: Plain CE (gamma=0) ==="
python3 run_patch_attack.py $COMMON \
  --ss_focal_gamma 0

echo ""
echo "All SS focal comparison runs finished! Check MLflow: ss_focal_comparison"
