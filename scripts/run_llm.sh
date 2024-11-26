#!/bin/bash

#SBATCH --job-name=eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --nodelist=fnlp-3090-52120

#SBATCH --chdir=/remote-home1/cktan/reps/sugar-crepe
#SBATCH --output=/remote-home1/cktan/reps/sugar-crepe/logs/eval.log

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)

model=/remote-home1/share/models/llama3_1_hf/Meta-Llama-3.1-8B
srun torchrun --nproc_per_node=${NUM_GPUS} /remote-home1/cktan/reps/sugar-crepe/eval_llm_parallel.py --model $model
