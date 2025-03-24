#!/bin/bash

# List of epsilon values to test
epsilons=("1./255." "2./255." "4./255." "8./255.")

# Loop over each epsilon value
for epsilon in "${epsilons[@]}"; do
  echo "Running CosPGD attack with epsilon=$epsilon and softmax enabled"
  python run_attack_ptlflow.py \
    --net raft \
    --attack CosPGD \
    --epsilon $(bc <<< "scale=5; $epsilon") \
    --alpha 0.01 \
    --steps 40 \
    --target zero \
    --save_iterations 3 5 10 15 20 25 30 35 40

  echo "Running PGD attack with epsilon=$epsilon"
  python run_attack_ptlflow.py \
    --net raft \
    --attack PGD \
    --epsilon $(bc <<< "scale=5; $epsilon") \
    --alpha 0.01 \
    --steps 40 \
    --target zero \
    --save_iterations 3 5 10 15 20 25 30 35 40

  echo "Running FGSM attack with epsilon=$epsilon"
  python run_attack_ptlflow.py \
    --net raft \
    --attack FGSM \
    --epsilon $(bc <<< "scale=5; $epsilon") \
    --alpha 0.01 \
    --steps 40 \
    --target zero \
    --save_iterations 3 5 10 15 20 25 30 35 40

  echo "Running MIFGSM attack with epsilon=$epsilon"
  python run_attack_ptlflow.py \
    --net raft \
    --attack MIFGSM \
    --epsilon $(bc <<< "scale=5; $epsilon") \
    --alpha 0.01 \
    --steps 40 \
    --target zero \
    --save_iterations 3 5 10 15 20 25 30 35 40

done