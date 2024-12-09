set -e
NUM_GPUS=8

weight_paths=($(find /root/checkpoints/midtune/llavanext-clip-vit-large-patch14-336-openai-Qwen2-0.5B-cmov3-tune_mlp_lastlayer-cmov3-midtune_vt-blip558k-epochs_3 -type d -name 'checkpoint*'))

model=/home/save_dir/cktan/models/clip-vit-large-patch14-336-openai
for weight_path in ${weight_paths[@]};do
    torchrun --nproc_per_node=${NUM_GPUS} --master_port=20001 ./eval_clip_parallel.py --model $model --weight_path $weight_path
done
# weight_path=/root/checkpoints/midtune/llavanext-clip-vit-large-patch14-336-openai-Qwen2-0.5B-tune_mlp-blip3kale558k-select_last_layer-midtune_vt_mlp-blip3kale3M-select_last_layer-CMO
# torchrun --nproc_per_node=${NUM_GPUS} --master_port=20001 ./eval_clip_parallel.py --model $model --weight_path $weight_path