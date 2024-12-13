NUM_GPUS=8

model=/root/checkpoints/projectors/llavanext-clip-vit-large-patch14-336-openai-Qwen2.5-0.5B-Instruct-tune_mlp-template-qwen_1_5-lr1e-3-bs256
torchrun --nproc_per_node=${NUM_GPUS} ./main_eval_parallel.py --model $model --model_base /home/save_dir/cktan/models/Qwen2.5-0.5B-Instruct --template qwen_1_5