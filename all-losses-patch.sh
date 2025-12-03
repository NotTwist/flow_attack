#!/bin/bash

# Список доступных лоссов
losses=("aee" "cosim" "mse" "focal" "huber" "charbonnier")

# Список весов
weights=(1.0 0.5 0.25 0.1)

for loss1 in "${losses[@]}"; do
  echo "Запуск с лоссами: $loss1"

  python run_patch_attack.py \
    --small_run \
    --target zero \
    --loss "${loss1}" \
    --model_name flownetc \
    --n 1\
    --save_artifacts
done
