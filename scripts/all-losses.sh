#!/bin/bash
cd "$(dirname "$0")/.."

# Список доступных лоссов
losses=("aee" "cosim" "mse" "focal" "huber" "charbonnier")

# Список весов
weights=(1.0 0.5 0.25 0.1)

for loss1 in "${losses[@]}"; do
  echo "Запуск с лоссами: $loss1"

  python3 run_attack_ptlflow.py \
    --small_run \
    --target zero \
    --saved_iterations 3 5 10 15 20 \
    --loss "${loss1}"
done
