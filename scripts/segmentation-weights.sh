#!/bin/bash
cd "$(dirname "$0")/.."
# constant epsilon (8/255)
EPSILON=$(bc  <<< "scale=8; 8/255")

# Define candidate weights
weights=(10.0 5.0 1.0 0.5 0.25 0.1 0.05 0.0)

for FLOW_W in "${weights[@]}"; do
  echo "Running attack with semantic-segmentation=$FLOW_W, epsilon=$EPSILON"

  python3 run_attack_ptlflow.py \
    --small_run \
    --target zero \
    --saved_iterations 3 5 10 15 20 \
    --attack_mde \
    --attack_ss \
    --ss_model pspnet_cityscapes \
    --ss_target targeted \
    --loss_weights 1.0 0.1 "$FLOW_W" 
done
