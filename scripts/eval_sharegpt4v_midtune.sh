NUM_GPUS=8

model=/root/checkpoints/midtune/llavanext-clip-vit-large-patch14-336-openai-Qwen2-0.5B-tune_mlp-select_last_layer-midtune_vt-sharegpt4v

torchrun --nproc_per_node=${NUM_GPUS} ./main_eval_parallel.py --model $model