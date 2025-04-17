#!/bin/bash

# List of epsilon values to test
epsilons=("1./255." "2./255." "4./255." "8./255.")

# Loop over each epsilon value
for epsilon in "${epsilons[@]}"; do
  echo "Running MIFGSM attack with no scaling and epsilon=$epsilon"
  python run_attack_ptlflow.py \
    --model_name raft \
    --attack_type MIFGSM \
    --epsilon $(bc <<< "scale=5; $epsilon") \
    --steps 40 \
    --target zero \
    --scaling_type none \
    --saved_iterations 3 5 10 15 20 25 30 35 40

  echo "Running MIFGSM attack with cospgd and epsilon=$epsilon"
  python run_attack_ptlflow.py \
    --model_name raft \
    --attack_type MIFGSM \
    --epsilon $(bc <<< "scale=5; $epsilon") \
    --steps 40 \
    --target zero \
    --scaling_type cospgd \
    --saved_iterations 3 5 10 15 20 25 30 35 40

  echo "Running MIFGSM attack with sobel and epsilon=$epsilon"
  python run_attack_ptlflow.py \
    --model_name raft \
    --attack_type MIFGSM \
    --epsilon $(bc <<< "scale=5; $epsilon") \
    --steps 40 \
    --target zero \
    --scaling_type sobel \
    --saved_iterations 3 5 10 15 20 25 30 35 40

  echo "Running MIFGSM attack with highfreq and epsilon=$epsilon"
  python run_attack_ptlflow.py \
    --model_name raft \
    --attack_type MIFGSM \
    --epsilon $(bc <<< "scale=5; $epsilon") \
    --steps 40 \
    --target zero \
    --scaling_type high_freq \
    --saved_iterations 3 5 10 15 20 25 30 35 40
done