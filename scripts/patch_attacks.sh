#!/bin/bash
cd "$(dirname "$0")/.."

python3 run_patch_attack.py  --attack_mde --patch_projection --attack_ss --patch_size 100 --loss_weights 0 0 1 --small_run --n 2 --y_scale 1
python3 run_patch_attack.py  --attack_mde --patch_projection --attack_ss --patch_size 100 --loss_weights 0 0 1 --small_run --n 2 --y_scale 3
python3 run_patch_attack.py  --attack_mde --patch_projection --attack_ss --patch_size 100 --loss_weights 0 0 1 --small_run --n 2 --y_scale 3 --tv_weight 1000 --nps_weight 1000



