# python ./main_eval_hf.py --model_base /remote-home1/share/models/Qwen/Qwen2.5-0.5B --model /remote-home1/cktan/reps/LLaVA-NeXT/checkpoints/projectors/llavanext-clip-vit-large-patch14-336-openai-Qwen2.5-0.5B-mlp2x_gelu-pretrain_blip558k_plain-bs64-lr1e-3
torchrun --nproc_per_node=2 ./main_eval_parallel.py --model /remote-home1/cktan/reps/LLaVA-NeXT/checkpoints/midtune/llavanext-qwen-midtune_vt-blip558k-epochs_5/checkpoint-43600