#!/bin/bash

# Список доступных лоссов
losses=("aee" "cosim" "mse" "epe" "focal" "huber" "charbonnier")

# Список весов
weights=(1.0 0.5 0.25 0.1)

for loss1 in "${losses[@]}"; do
  for loss2 in "${losses[@]}"; do
    # исключаем одинаковые пары (aee,aee и т.п.)
    if [[ "$loss1" == "$loss2" ]]; then
      continue
    fi

    for w1 in "${weights[@]}"; do
      for w2 in "${weights[@]}"; do
        echo "Запуск с лоссами: $loss1:$w1, $loss2:$w2"

        python run_attack_ptlflow.py \
          --small_run \
          --target zero \
          --saved_iterations 3 5 10 15 20 \
          --loss "${loss1}:${w1}" "${loss2}:${w2}"
      done
    done
  done
done
