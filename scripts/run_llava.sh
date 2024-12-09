NUM_GPUS=8

model=($(find /root/checkpoints/midtune/llavanext-clip-vit-large-patch14-336-openai-Qwen2-0.5B-cmov2-tune_mlp_lastlayer-cmov2-midtune_vt-blip558k-epochs_5 -type d -name 'checkpoint*'))
for m in ${model[@]};do
    torchrun --nproc_per_node=${NUM_GPUS} ./main_eval_parallel.py --model $m
done