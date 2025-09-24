#!/bin/bash

# List of epsilon values to test
epsilons=("8./255.")

# Loop over each epsilon value
for epsilon in "${epsilons[@]}"; do
  # echo "Running FGSM attack with aee loss epsilon=$epsilon"
  # python run_attack_ptlflow.py \
  #   --model_name raft \
  #   --attack_type FGSM \
  #   --epsilon $(bc <<< "scale=5; $epsilon") \
  #   --steps 20 \
  #   --target zero \
  #   --small_run \
  #   --saved_iterations 3 5 10 15 20

  # echo "Running FGSM attack with huber loss epsilon=$epsilon"
  # python run_attack_ptlflow.py \
  #   --model_name raft \
  #   --attack_type FGSM \
  #   --epsilon $(bc <<< "scale=5; $epsilon") \
  #   --steps 20 \
  #   --target zero \
  #   --small_run \
  #   --loss huber \
  #   --saved_iterations 3 5 10 15 20

  echo "Running FGSM attack with charbonnier loss epsilon=$epsilon"
  python run_attack_ptlflow.py \
    --model_name raft \
    --attack_type FGSM \
    --epsilon $(bc <<< "scale=5; $epsilon") \
    --steps 20 \
    --target zero \
    --small_run \
    --loss focal \
    --saved_iterations 3 5 10 15 20
done