#!/bin/bash

# List of epsilon values to test
epsilons=("1./255." "2./255." "4./255." "8./255.")

# Loop over each epsilon value
for epsilon in "${epsilons[@]}"; do
  echo "Running CosPGD attack epsilon=$epsilon"
  python run_attack_ptlflow.py \
    --model_name raft \
    --attack_type CosPGD \
    --epsilon $(bc <<< "scale=5; $epsilon") \
    --alpha 0.01 \
    --steps 40 \
    --target zero \
    --saved_iterations 3 5 10 15 20 25 30 35 40

  echo "Running Sobel attack epsilon=$epsilon"
  python run_attack_ptlflow.py \
    --model_name raft \
    --attack_type Sobel \
    --epsilon $(bc <<< "scale=5; $epsilon") \
    --alpha 0.01 \
    --steps 40 \
    --target zero \
    --saved_iterations 3 5 10 15 20 25 30 35 40


  echo "Running HighFreq attack epsilon=$epsilon"
  python run_attack_ptlflow.py \
    --model_name raft \
    --attack_type HighFreq \
    --epsilon $(bc <<< "scale=5; $epsilon") \
    --alpha 0.01 \
    --steps 40 \
    --target zero \
    --saved_iterations 3 5 10 15 20 25 30 35 40

done