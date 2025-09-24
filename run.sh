#!/bin/bash

# List of epsilon values to test
epsilons=("1./255." "2./255." "4./255." "8./255.")

# Loop over each epsilon value
for epsilon in "${epsilons[@]}"; do
  echo "Running NIFGSM attack with epsilon=$epsilon"
  python run_attack_ptlflow.py \
    --model_name raft \
    --attack_type NIFGSM \
    --epsilon $(bc <<< "scale=5; $epsilon") \
    --steps 40 \
    --target zero \
    --saved_iterations 3 5 10 15 20 25 30 35 40

  echo "Running PIFGSM attack with epsilon=$epsilon"
  python run_attack_ptlflow.py \
    --model_name raft \
    --attack_type PIFGSM \
    --epsilon $(bc <<< "scale=5; $epsilon") \
    --steps 40 \
    --target zero \
    --saved_iterations 3 5 10 15 20 25 30 35 40

  echo "Running EMIFGSM attack with epsilon=$epsilon"
  python run_attack_ptlflow.py \
    --model_name raft \
    --attack_type EMIFGSM \
    --epsilon $(bc <<< "scale=5; $epsilon") \
    --steps 40 \
    --target zero \
    --saved_iterations 3 5 10 15 20 25 30 35 40

  echo "Running VMIFGSM attack with epsilon=$epsilon"
  python run_attack_ptlflow.py \
    --model_name raft \
    --attack_type VMIFGSM \
    --epsilon $(bc <<< "scale=5; $epsilon") \
    --steps 40 \
    --target zero \
    --saved_iterations 3 5 10 15 20 25 30 35 40

  echo "Running VNIFGSM attack with epsilon=$epsilon"
  python run_attack_ptlflow.py \
    --model_name raft \
    --attack_type VNIFGSM \
    --epsilon $(bc <<< "scale=5; $epsilon") \
    --steps 40 \
    --target zero \
    --saved_iterations 3 5 10 15 20 25 30 35 40
done