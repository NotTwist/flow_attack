#!/bin/bash
cd "$(dirname "$0")/.."

# Evaluate the best patches from each weight-strategy training run.
# All three share identical base config; only the patch file differs.

COMMON="--attack_mde --attack_ss --patch_projection --patch_size 100 --y_scale 3 --n 1 --steps 1 --experiment_name weight_strategy_eval"

PATCH_FIXED="experiment_data/patch_train_raft_patch_Kitti15_2026-03-23_12-54-22_2.png"
PATCH_NORMALIZED="experiment_data/patch_train_raft_patch_Kitti15_2026-03-24_01-03-16_5.png"
PATCH_MINMAX="experiment_data/patch_train_raft_patch_Kitti15_2026-03-24_06-42-14_5.png"

echo "=== Eval: fixed weights ==="
python3 run_patch_attack.py $COMMON --trained_patch "$PATCH_FIXED"

echo ""
echo "=== Eval: normalized weights ==="
python3 run_patch_attack.py $COMMON --trained_patch "$PATCH_NORMALIZED"

echo ""
echo "=== Eval: minmax weights ==="
python3 run_patch_attack.py $COMMON --trained_patch "$PATCH_MINMAX"

echo ""
echo "All eval runs finished! Check MLflow experiment: weight_strategy_eval"
