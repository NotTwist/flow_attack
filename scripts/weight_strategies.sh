#!/bin/bash
cd "$(dirname "$0")/.."

# Compare the three loss-weight strategies (fixed / normalized / minmax)
# across multi-task patch attacks on OF + MDE + SS.
#
# All runs use the same base config so results are directly comparable.
# Logged to mlflow under a shared experiment name for easy comparison.

COMMON="--attack_mde --attack_ss --patch_projection --target down --patch_size 100 --y_scale 3 --n 5 --steps 20 --experiment_name weight_strategy_comparison"

# ── 1. Fixed (equal weights) ────────────────────────────────────────
# echo "=== Fixed: equal weights (1/3 each) ==="
# python3 run_patch_attack.py $COMMON \
#   --weight_strategy fixed \
#   --loss_weights 1.0 1.0 1.0

# ── 2. Fixed (hand-tuned weights) ───────────────────────────────────
# echo "=== Fixed: hand-tuned (1.0 / 0.1 / 1.0) ==="
# python3 run_patch_attack.py $COMMON \
#   --weight_strategy fixed \
#   --loss_weights 1.0 0.1 1.0

# ── 3. Normalized (1/L_clean) ───────────────────────────────────────
echo "=== Normalized: wi = user_w / Li(clean) ==="
python3 run_patch_attack.py $COMMON \
  --weight_strategy normalized \
  --loss_weights 1.0 1.0 1.0

# ── 4. Min-max (APGDA, default hypers) ─────────────────────────────
echo "=== Min-max: alpha_w=0.03, gamma=5.0 ==="
python3 run_patch_attack.py $COMMON \
  --weight_strategy minmax \
  --minmax_alpha_w 0.03 \
  --minmax_gamma 5.0

# ── 5. Min-max (higher weight lr) ──────────────────────────────────
echo "=== Min-max: alpha_w=0.1, gamma=5.0 ==="
python3 run_patch_attack.py $COMMON \
  --weight_strategy minmax \
  --minmax_alpha_w 0.1 \
  --minmax_gamma 5.0

# ── 6. Min-max (stronger regularisation towards uniform) ────────────
echo "=== Min-max: alpha_w=0.03, gamma=10.0 ==="
python3 run_patch_attack.py $COMMON \
  --weight_strategy minmax \
  --minmax_alpha_w 0.03 \
  --minmax_gamma 10.0

echo "All weight-strategy runs finished!"
