#!/bin/bash

# --------------------------------------------
#  Настройки эксперимента
# --------------------------------------------

# Epsilon значения
epsilons=("8./255.")

# Список моделей, которые ты перечислил
models=(
  "pwcnet"
  "flowformer"
  "gma"
  "raft"
  "sea_raft"
  "videoflow_bof"
  "memflow"
  "flow1d"
  "meflow"
  "rpknet"
)

# Список атак из твоего задания
attacks=(
  "FGSM"
  "PGD"
  "MIFGSM"
  "CosPGD"
  "APGD"
  "ADAMIFGSM"
  "EMIFGSM"
  "VMIFGSM"
  "VNIFGSM"
)

# Итерации, которые сохраняем для метрик
saved_iter="3 5 10 15 20"

# Количество шагов для итеративных атак
steps=20

# Общие параметры
dataset="Kitti15"
target="zero"

experiment_name="flow_benchmark"
# --------------------------------------------
#  Основной цикл
# --------------------------------------------

for epsilon in "${epsilons[@]}"; do
  # вычисляем числовое значение через bc
  eps_val=$(bc <<< "scale=5; $epsilon")

  for model in "${models[@]}"; do
    echo "=============================================="
    echo "Model: $model  |  Epsilon: $eps_val"
    echo "=============================================="

    for attack in "${attacks[@]}"; do
      echo "--- Running $attack ---"

      # Каталог для сохранения результатов
      outdir="/mnt/ssd1/28s_mur/results/${model}/${attack}/eps_${epsilon//\//_}"
      mkdir -p "$outdir"

      python run_attack_ptlflow.py \
        --model_name "$model" \
        --attack_type "$attack" \
        --epsilon "$eps_val" \
        --steps "$steps" \
        --target "$target" \
        --dataset "$dataset" \
        --saved_iterations $saved_iter \
        --output_dir "$outdir" \
        --experiment_name "$experiment_name"
      echo "Completed: $model | $attack | eps=${eps_val}"
      echo
      sleep 1
    done
  done
done

echo "All experiments finished!"
