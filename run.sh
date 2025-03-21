#!/bin/bash

# List of epsilon values to test
epsilons=("0.0039" "0.0078" "0.0156" "0.0313")

# Loop over each epsilon value
for epsilon in "${epsilons[@]}"; do
  echo "Running CosPGD attack with epsilon=$epsilon and softmax enabled"
  python run_attack.py \
    --net PWCNet \
    --attack CosPGD \
    --epsilon $epsilon \
    --steps 20 \
    --target neg_flow \
    --dataset Kitti15 \
    --save_artifacts \
    --small_run
  
  echo "Running CosPGD attack with epsilon=$epsilon and softmax disabled"
  python run_attack.py \
    --net PWCNet \
    --attack CosPGD \
    --epsilon $epsilon \
    --steps 20 \
    --target neg_flow \
    --dataset Kitti15 \
    --save_artifacts \
    --no_softmax \
    --small_run
done
