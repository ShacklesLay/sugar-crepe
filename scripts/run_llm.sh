NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)

model=/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/pywang/cktan/models/Qwen2.5-3B-Instruct
torchrun --nproc_per_node=${NUM_GPUS} ./eval_llm_parallel.py --model $model
