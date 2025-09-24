#!/bin/bash
# constant epsilon (8/255)
EPSILON=$(bc  <<< "scale=8; 8/255")

# Define candidate weights
weights=(1.0 0.75 0.5 0.25 0.1 0.05)

for FLOW_W in "${weights[@]}"; do
  for MDE_W in "${weights[@]}"; do
    echo "Running attack with optical-flow=$FLOW_W, mde=$MDE_W, epsilon=$EPSILON"

    python run_attack_ptlflow.py \
      --small_run \
      --target zero \
      --saved_iterations 3 5 10 15 20 \
      --attack_mde \
      --loss_weights "$FLOW_W" "$MDE_W"
  done
done
