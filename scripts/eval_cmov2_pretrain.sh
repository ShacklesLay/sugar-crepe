NUM_GPUS=8

# model=($(find /home/image_data/cktan/reps/LLaVA-NeXT/checkpoints/midtune/llavanext-clip-vit-large-patch14-336-openai-Qwen2-0.5B-tune_mlp-blip3kale558k-select_last_layer-midtune_vt_mlp-blip3kale3M-select_last_layer -type d -name 'checkpoint*'))
# for m in ${model[@]};do
#     torchrun --nproc_per_node=${NUM_GPUS} ./main_eval_parallel.py --model $m
# done

model=/root/checkpoints/projectors/llavanext-clip-vit-large-patch14-336-openai-Qwen2-0.5B-cmov2-tune_mlp_lastlayer_lr1e-3_bs64
torchrun --nproc_per_node=${NUM_GPUS} ./main_eval_parallel.py --model $model --model_base /home/save_dir/cktan/models/Qwen2-0.5B