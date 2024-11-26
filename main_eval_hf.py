import argparse
import json
import os
import os.path as osp

import torch
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm
import copy

import sys
sys.path.append("/remote-home1/cktan/server_tools/")
from scripts import dump, load, get_rank_and_world_size
from larknotice import lark_sender

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.conversation import conv_templates, SeparatorStyle
DEFAULT_IMAGE_TOKEN = '<image>'
IMAGE_TOKEN_INDEX = -200

def get_ppl(text, image, model, tokenizer, image_processor, conv_template, verbose=False):
    content = DEFAULT_IMAGE_TOKEN + '\n'
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.bfloat16, device='cuda') for _image in image_tensor]
    
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], content)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    prompt_question += text
    
    if verbose:
        print(prompt_question)
    
    inputs_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
    
    cont = model(inputs_ids, images=image_tensor, labels=inputs_ids, return_dict=True)
    return cont.loss.item()

@torch.no_grad()
def text_retrieval(pos_text, neg_text, image, model, tokenizer, image_processor, mode, verbose):
    # import ipdb; ipdb.set_trace()
    if mode == 'ppl':
        conv_template = 'plain'
        pos_ppl = get_ppl(pos_text, image, model, tokenizer, image_processor, conv_template, verbose)
        neg_ppl = get_ppl(neg_text, image, model, tokenizer, image_processor, conv_template)
        return 1 if pos_ppl < neg_ppl else 0
    return 
    
    
def evaluate(image_root, dataset, model, tokenizer, image_processor, mode):
    metrics = {}
    for c, data_dict in dataset.items():
        correct_cnt=0
        for i, data in tqdm(data_dict.items(), desc=f'evaluating {c}'):
            verbose = True if i==0 else False
            image_path = os.path.join(image_root, data['filename'])
            image = Image.open(image_path).convert('RGB')
            correct = text_retrieval(data['caption'], data['negative_caption'], image, model, tokenizer, image_processor, mode, verbose)
            correct_cnt += correct
        count = len(data_dict)
        metrics[c] = correct_cnt / count
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default= "/remote-home1/cktan/reps/LLaVA-NeXT/checkpoints/midtune/llavanext-clip-vit-large-patch14-336-openai-Qwen2.5-0.5B-mlp2x_gelu-midtune_blip558k_plain-tune_vt")
    parser.add_argument('--output', type=str, default='./outputs', help="Directory to where results are saved")
    parser.add_argument('--coco_image_root', type=str, default='/remote-home1/share/data/COCO2017val/val2017')
    parser.add_argument('--model_base', type=str, default=None)
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--mode', type=str, default='ppl')

    args = parser.parse_args()

    model_path = args.model

    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path,args.model_base, model_name,torch_dtype="bfloat16")
    model.eval()
    
    # conv_template="qwen_1_5"
    conv_template = 'plain'

    data_dict = {
        'add_obj'    : f'{args.data_root}/add_obj.json',
        'add_att'    : f'{args.data_root}/add_att.json',
        'replace_obj': f'{args.data_root}/replace_obj.json',
        'replace_att': f'{args.data_root}/replace_att.json',
        'replace_rel': f'{args.data_root}/replace_rel.json',
        'swap_obj'   : f'{args.data_root}/swap_obj.json',
        'swap_att'   : f'{args.data_root}/swap_att.json',
    }
    dataset = {}
    for c, data_path in data_dict.items():
        dataset[c] = json.load(open(data_path, 'r', encoding='utf-8'))
    
    os.makedirs(args.output, exist_ok=True)
    
    model_name = model_name.split('/')[-1]
    metrics = evaluate(args.coco_image_root, dataset, model, tokenizer, image_processor, args.mode)
    print(metrics)
    print(f"Dump results to: {os.path.join(args.output, f'{model_name}.json')}")
    json.dump(metrics, open(os.path.join(args.output, f'{model_name}.json'), 'w'), indent=4)