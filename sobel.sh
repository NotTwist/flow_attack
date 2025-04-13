#!/bin/bash

# List of epsilon values to test
epsilons=("1./255." "2./255." "4./255." "8./255.")

# Loop over each epsilon value
for epsilon in "${epsilons[@]}"; do
  echo "Running Sobel attack with map scaling epsilon=$epsilon"
  python run_attack_ptlflow.py \
    --model_name raft \
    --attack_type Sobel \
    --epsilon $(bc <<< "scale=5; $epsilon") \
    --alpha 0.01 \
    --steps 20 \
    --target zero \
    --small_run \
    --saved_iterations 3 5 10 15 20


  echo "Running HighFreq attack with map scaling epsilon=$epsilon"
  python run_attack_ptlflow.py \
    --model_name raft \
    --attack_type HighFreq \
    --epsilon $(bc <<< "scale=5; $epsilon") \
    --alpha 0.01 \
    --steps 20 \
    --target zero \
    --small_run \
    --saved_iterations 3 5 10 15 20
done