#!/bin/bash
python run_patch_attack.py --small_run --model_name flownetc --n 10 --attack_mde --loss_weights 1 0.1
python run_patch_attack.py --small_run --model_name flownetc --n 10 --attack_mde --loss_weights 0 0.1
python run_patch_attack.py --small_run --model_name flownetc --n 10 --attack_mde --loss_weights 1 0


