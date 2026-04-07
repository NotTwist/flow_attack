#!/bin/bash
cd "$(dirname "$0")/.."

# List of epsilon values to test
epsilons=("1./255." "2./255." "4./255." "8./255.")
steps=( 3 5 10 15 20 25 30 35 40)
# Loop over each epsilon value
for epsilon in "${epsilons[@]}"; do
for step in "${steps[@]}"; do
  echo "Running APGD attack with epsilon=$epsilon and steps=$step"
  python3 run_attack_ptlflow.py \
    --net raft \
    --attack APGD \
    --epsilon $(bc <<< "scale=5; $epsilon") \
    --alpha 0.01 \
    --target zero \
    --steps $step
done
done