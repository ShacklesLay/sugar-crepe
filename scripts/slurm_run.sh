#!/bin/bash

#SBATCH --job-name=eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G

#SBATCH --chdir=/remote-home1/cktan/reps/sugar-crepe
#SBATCH --output=/remote-home1/cktan/reps/sugar-crepe/logs/eval.log

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)

model=($(find /remote-home1/cktan/reps/LLaVA-NeXT/checkpoints/midtune/blip3kale/llavanext-qwen-midtune_vt-blip3kale10m-all -type d -name 'checkpoint*'))
for m in ${model[@]};do
    torchrun --nproc_per_node=${NUM_GPUS} ./main_eval_parallel.py --model $m
done