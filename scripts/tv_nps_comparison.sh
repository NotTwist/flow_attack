#!/bin/bash
cd "$(dirname "$0")/.."

# Compare attack performance with and without TV / NPS regularization.
# TV raw magnitude ~0.01-0.1, NPS raw magnitude ~0.02-0.05,
# adversarial loss ~1-50. Weight of 100-1000 brings them to same scale.

COMMON="--attack_mde --attack_ss --patch_projection --target down --patch_size 100 --y_scale 3 --n 5 --steps 10 --experiment_name tv_nps_comparison"

echo "=== 1/5: No regularization ==="
python3 run_patch_attack.py $COMMON \
  --tv_weight 0 --nps_weight 0

echo ""
echo "=== 2/5: TV=100, NPS=100 (light) ==="
python3 run_patch_attack.py $COMMON \
  --tv_weight 100 --nps_weight 100

echo ""
echo "=== 3/5: TV=1000, NPS=1000 (strong) ==="
python3 run_patch_attack.py $COMMON \
  --tv_weight 1000 --nps_weight 1000

echo ""
echo "=== 4/5: TV only (1000) ==="
python3 run_patch_attack.py $COMMON \
  --tv_weight 1000 --nps_weight 0

echo ""
echo "=== 5/5: NPS only (1000) ==="
python3 run_patch_attack.py $COMMON \
  --tv_weight 0 --nps_weight 1000

echo ""
echo "All TV/NPS comparison runs finished! Check MLflow: tv_nps_comparison"
