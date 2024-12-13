set -e
NUM_GPUS=8
model=/home/save_dir/cktan/models/clip-vit-large-patch14-336-openai

# weight_paths=($(find /root/checkpoints/midtune/llavanext-clip-vit-large-patch14-336-openai-Qwen2-0.5B-cmov3-tune_mlp_lastlayer-cmov3-midtune_vt-blip558k-epochs_3 -type d -name 'checkpoint*'))

# for weight_path in ${weight_paths[@]};do
#     torchrun --nproc_per_node=${NUM_GPUS} --master_port=20001 ./eval_clip_parallel.py --model $model --weight_path $weight_path
# done
weight_path=/root/checkpoints/midtune/llavanext-clip-vit-large-patch14-336-openai-Qwen2-0.5B-cmov3-tune_mlp_lastlayer-cmov3-midtune_vt-blip558k-epochs_10/checkpoint-87200
torchrun --nproc_per_node=${NUM_GPUS} --master_port=20001 ./eval_clip_parallel.py --model $model --weight_path $weight_path