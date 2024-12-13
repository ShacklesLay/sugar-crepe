NUM_GPUS=8
model=($(find /root/checkpoints/midtune/llavanext-clip-vit-large-patch14-336-openai-Qwen2-0.5B-tune_mlp-select_last_layer-cmov2token-midtune_vt-blip558k-epochs_3 -type d -name 'checkpoint*'))
for m in ${model[@]};do
    torchrun --nproc_per_node=${NUM_GPUS} ./main_eval_parallel.py --model $m
done