NUM_GPUS=8

model=/home/save_dir/cktan/models/clip-vit-large-patch14-336-openai
torchrun --nproc_per_node=${NUM_GPUS} ./eval_clip_parallel.py --model $model
