#!/bin/bash
cd "$(dirname "$0")/.."

# List of epsilon values to test
epsilons=("1./255." "2./255." "4./255." "8./255.")

# Loop over each epsilon value
for epsilon in "${epsilons[@]}"; do
  echo "Running GradCAM attack target_layer=cnet.conv2 with epsilon=$epsilon"
  python3 run_attack_ptlflow.py \
    --model_name raft \
    --attack_type GradCAM \
    --epsilon $(bc <<< "scale=5; $epsilon") \
    --alpha 0.01 \
    --steps 20 \
    --target zero \
    --target_layer cnet.conv2 \
    --small_run \
    --use_map_scaling \
    --saved_iterations 3 5 10 15 20

  echo "Running GradCAM attack target_layer=update_block.flow_head with epsilon=$epsilon"
  python3 run_attack_ptlflow.py \
    --model_name raft \
    --attack_type GradCAM \
    --epsilon $(bc <<< "scale=5; $epsilon") \
    --alpha 0.01 \
    --steps 20 \
    --target zero \
    --target_layer update_block.flow_head \
    --small_run \
    --use_map_scaling \
    --saved_iterations 3 5 10 15 20

  echo "Running GradCAM attack target_layer=update_block.mask with epsilon=$epsilon"
  python3 run_attack_ptlflow.py \
    --model_name raft \
    --attack_type GradCAM \
    --epsilon $(bc <<< "scale=5; $epsilon") \
    --alpha 0.01 \
    --steps 20 \
    --target zero \
    --target_layer update_block.mask \
    --small_run \
    --use_map_scaling \
    --saved_iterations 3 5 10 15 20
done