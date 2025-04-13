#!/bin/bash

# List of epsilon values to test
alphas=( "0.005" "0.01" "0.02" "0.04" "0.1")
# Loop over each epsilon value
for alpha in "${alphas[@]}"; do
  echo "Running CosPGD attack with alpha=$alpha and softmax enabled"
  python run_attack_ptlflow.py \
    --model_name raft \
    --attack_type CosPGD \
    --alpha $alpha \
    --steps 40 \
    --target zero \
    --small_run \
    --saved_iterations 3 5 10 15 20 25 30 35 40

  echo "Running PGD attack with alpha=$alpha"
  python run_attack_ptlflow.py \
    --model_name raft \
    --attack_type PGD \
    --alpha $alpha \
    --steps 40 \
    --target zero \
    --small_run \
    --saved_iterations 3 5 10 15 20 25 30 35 40

  echo "Running FGSM attack with alpha=$alpha"
  python run_attack_ptlflow.py \
    --model_name raft \
    --attack_type FGSM \
    --alpha $alpha \
    --steps 40 \
    --target zero \
    --small_run \
    --saved_iterations 3 5 10 15 20 25 30 35 40

  echo "Running MIFGSM attack with alpha=$alpha"
  python run_attack_ptlflow.py \
    --model_name raft \
    --attack_type MIFGSM \
    --alpha $alpha \
    --steps 40 \
    --target zero \
    --small_run \
    --saved_iterations 3 5 10 15 20 25 30 35 40

done