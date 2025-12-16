#!/bin/bash

# List of epsilon values to test
epsilons=("8./255.")

# Loop over each epsilon value
for epsilon in "${epsilons[@]}"; do
  echo "Running MIFGSM attack with epsilon=$epsilon"
  python run_attack_ptlflow.py \
    --model_name raft \
    --attack_type MIFGSM \
    --epsilon $(bc <<< "scale=5; $epsilon") \
    --steps 20 \
    --target zero \
    --saved_iterations 3 5 10 15 20 
    
  echo "Running MIFGSM attack with epsilon=$epsilon"
  python run_attack_ptlflow.py \
    --model_name raft \
    --attack_type MIFGSM \
    --epsilon $(bc <<< "scale=5; $epsilon") \
    --steps 20 \
    --target zero \
    --saved_iterations 3 5 10 15 20 

  echo "Running EMIFGSM attack with epsilon=$epsilon"
  python run_attack_ptlflow.py \
    --model_name raft \
    --attack_type EMIFGSM \
    --epsilon $(bc <<< "scale=5; $epsilon") \
    --steps 20 \
    --target zero \
    --saved_iterations 3 5 10 15 20

  echo "Running VMIFGSM attack with epsilon=$epsilon"
  python run_attack_ptlflow.py \
    --model_name raft \
    --attack_type VMIFGSM \
    --epsilon $(bc <<< "scale=5; $epsilon") \
    --steps 20 \
    --target zero \
    --saved_iterations 3 5 10 15 20

  echo "Running VNIFGSM attack with epsilon=$epsilon"
  python run_attack_ptlflow.py \
    --model_name raft \
    --attack_type VNIFGSM \
    --epsilon $(bc <<< "scale=5; $epsilon") \
    --steps 20 \
    --target zero \
    --saved_iterations 3 5 10 15 20
done