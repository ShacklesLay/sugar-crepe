NUM_GPUS=8

model=/home/save_dir/cktan/models/Qwen2-0.5B
torchrun --nproc_per_node=${NUM_GPUS} ./eval_llm_parallel.py --model $model
