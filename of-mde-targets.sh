#!/bin/bash

echo "zero zero"
python run_attack_ptlflow.py --small_run --attack_mde --attack_type PGD --target zero --mde_target zero --saved_iterations 3 5 10 15 20 --loss_weights 1 0
echo "untargeted zero"
python run_attack_ptlflow.py --small_run --attack_mde --attack_type PGD --target untargeted --mde_target zero --saved_iterations 3 5 10 15 20 --loss_weights 1 0
echo "zero untargeted"
python run_attack_ptlflow.py --small_run --attack_mde --attack_type PGD --target zero --mde_target untargeted --saved_iterations 3 5 10 15 20 --loss_weights 0 1
echo "untargeted untargeted"
python run_attack_ptlflow.py --small_run --attack_mde --attack_type PGD --target untargeted --mde_target untargeted --saved_iterations 3 5 10 15 20 --loss_weights 0 1