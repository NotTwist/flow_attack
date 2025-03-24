#!/bin/bash

# List of epsilon values to test
epsilons=("1./255." "2./255." "4./255." "8./255.")

# Loop over each epsilon value
for epsilon in "${epsilons[@]}"; do
  echo "Running CosPGD attack with epsilon=$epsilon and softmax enabled"
  python run_attack.py \
    --net RAFT \
    --attack CosPGD \
    --epsilon $(bc <<< "scale=5; $epsilon") \
    --alpha 0.01 \
    --steps 20 \
    --target neg_flow \
    --dataset Sintel \
    --save_iterations 3 5 10 15 20
  
  echo "Running CosPGD attack with epsilon=$epsilon and softmax disabled"
  python run_attack.py \
    --net RAFT \
    --attack CosPGD \
    --epsilon $(bc <<< "scale=5; $epsilon") \
    --alpha 0.01 \
    --steps 20 \
    --target neg_flow \
    --dataset Sintel \
    --save_iterations 3 5 10 15 20 \
    --no_softmax 

  echo "Running PGD attack with epsilon=$epsilon"
  python run_attack.py \
    --net RAFT \
    --attack PGD \
    --epsilon $(bc <<< "scale=5; $epsilon") \
    --alpha 0.01 \
    --steps 20 \
    --target neg_flow \
    --dataset Sintel \
    --save_iterations 3 5 10 15 20

  echo "Running FGSM attack with epsilon=$epsilon"
  python run_attack.py \
    --net RAFT \
    --attack FGSM \
    --epsilon $(bc <<< "scale=5; $epsilon") \
    --alpha 0.01 \
    --steps 20 \
    --target neg_flow \
    --dataset Sintel \
    --save_iterations 3 5 10 15 20

done