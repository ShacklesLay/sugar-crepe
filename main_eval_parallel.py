import argparse
import json
import os
import os.path as osp
from PIL import Image
from tqdm import tqdm
import re
import copy
import pandas as pd
import datetime

import torch
import torch.distributed as dist

import sys
sys.path.append("../server_tools")
from scripts import dump, load, get_rank_and_world_size

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
DEFAULT_IMAGE_TOKEN = '<image>'
IMAGE_TOKEN_INDEX = -200
IGNORE_INDEX = -100

def preprocess_qwen(sources, tokenizer, has_image: bool = False, max_len=2048, system_message = "You are a helpful assistant."):
    for source in sources:
        for sentence in source:
            num_im = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence['value']))
            if num_im == 1 and DEFAULT_IMAGE_TOKEN in sentence['value'] and not sentence['value'].startswith(DEFAULT_IMAGE_TOKEN):
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
                sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
                sentence["value"] = sentence["value"].strip()
    
    # roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}
    roles = {"human": "user", "gpt": "assistant"}

    # Add image tokens to tokenizer as a special tokens
    # When there is actually an image, we add the image tokens as a special token
    if has_image:
        tokenizer.add_tokens(["<image>"], special_tokens=True)

    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    # unmask_tokens = ["<|im_start|>", "<|im_start|>", "\n"]
    unmask_tokens_idx =  [198, 151644, 151645]
    nl_tokens = tokenizer("\n").input_ids

    # Reset Qwen chat templates so that it won't include system message every time we apply
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    # _system = tokenizer("system").input_ids + nl_tokens
    # _user = tokenizer("user").input_ids + nl_tokens
    # _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        # New version, use apply chat template
        # Build system message for each sentence
        input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            # Make sure llava data can load
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role =  roles.get(role, role)
            
            conv = [{"role" : role, "content" : content}]
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id
        

                    
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX
        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )

def get_ppl(text, image, model, tokenizer, image_processor, template='plain', verbose=False):
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.bfloat16, device='cuda') for _image in image_tensor]

    if template == 'plain':
        prompt_question = DEFAULT_IMAGE_TOKEN + text + '\n'
        inputs_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
        labels = inputs_ids.clone()
        if verbose:
            print(prompt_question)  
        
    elif template == 'qwen_1_5':
        source = [
            {
                'from': 'human',
                'value':'Write a terse but informative summary of the picture.\n<image>',
            },
            {
                'from': 'gpt',
                'value': text,
            }]
        data_dict = preprocess_qwen([source], tokenizer, has_image=True, system_message="You are a helpful assistant.")
        inputs_ids = data_dict['input_ids'].cuda()
        labels = data_dict['labels'].cuda()
        if verbose:
            tokenizer = copy.deepcopy(tokenizer)
            tokenizer.add_tokens(["<image>"], special_tokens=True)
            input_ids = []
            for i in data_dict['input_ids'][0]:
                if i != IMAGE_TOKEN_INDEX:
                    input_ids.append(i)
                else:
                    input_ids.append(tokenizer.convert_tokens_to_ids("<image>"))
            print(f"Template:\n {tokenizer.decode(input_ids)}")
      
    cont = model(inputs_ids, images=image_tensor, labels=labels, return_dict=True)
    return cont.loss.item()


def infer_data(model, tokenizer, image_processor, work_dir, dataset, out_file, dataset_name, image_root, template):
    # Load previous results
    prev_file = f'{work_dir}/{dataset_name}_PREV.pkl'
    res = load(prev_file) if osp.exists(prev_file) else {}
    if osp.exists(out_file):
        res.update(load(out_file))
    
    # Split the dataset for parallel processing
    rank, world_size = get_rank_and_world_size()
    sheet_indices = list(range(rank, len(dataset), world_size))
    lt = len(sheet_indices)
    data = dataset.iloc[sheet_indices]
    data_indices = [i for i in data['index']]
    
    # Check if all data has been inferred
    all_finished = True
    for i in range(lt):
        idx = data.iloc[i]['index']
        if idx not in res:
            all_finished = False
            break
    if all_finished:
        res = {k: res[k] for k in data_indices}
        dump(res, out_file)
        return
    
    # Data need to be inferred
    data = data[~data['index'].isin(res)]
    lt = len(data)
    
    for i in tqdm(range(lt)):
        idx = data.iloc[i]['index']
        if idx in res:
            continue
        
        # Core inference logic
        image_path = os.path.join(image_root, data.iloc[i]['image_path'])
        image = Image.open(image_path).convert('RGB')
        pos_text = data.iloc[i]['caption']
        neg_text = data.iloc[i]['negative_caption']
        pos_ppl = get_ppl(pos_text, image, model, tokenizer, image_processor, template)
        neg_ppl = get_ppl(neg_text, image, model, tokenizer, image_processor, template)
        torch.cuda.empty_cache()
        
        res[idx] = {
            'pos_ppl': pos_ppl,
            'neg_ppl': neg_ppl,
            'prediction': 1 if pos_ppl < neg_ppl else 0
        }
        
        if i % 100 == 0:
            dump(res, out_file)
    
    res = {k: res[k] for k in data_indices}
    dump(res, out_file)
        

# A wrapper for infer_data, do the pre & post processing
def infer_data_job(dataset, model, image_root, tokenizer, image_processor, dataset_name, work_dir, template):
    rank, world_size = get_rank_and_world_size()
    result_file = osp.join(work_dir, f'{dataset_name}.xlsx')
    
    # Check if the result file exists，if exists, load the result and dump it to a new file
    prev_file = f'{work_dir}/{dataset_name}_PREV.pkl'
    if osp.exists(result_file):
        if rank == 0:
            data = load(result_file)
            results = data.to_dict(orient='index')
            dump(results, prev_file)
        if world_size > 1:
            dist.barrier()
            
    tmpl = osp.join(work_dir, '{}' + f'{world_size}_{dataset_name}.pkl')
    out_file = tmpl.format(rank)

    infer_data(
        model, tokenizer, image_processor, work_dir=work_dir, dataset=dataset, out_file=out_file, dataset_name=dataset_name, image_root=image_root, template=template)
    if world_size > 1:
        dist.barrier()

    # Merge the results and dump it to the final file
    if rank == 0:
        data_all = {}
        for i in range(world_size):
            data_all.update(load(tmpl.format(i)))

        data = dataset
        for x in data['index']:
            assert x in data_all
        
        data['pos_ppl'] = [str(data_all[x]['pos_ppl']) for x in data['index']]
        data['neg_ppl'] = [str(data_all[x]['neg_ppl']) for x in data['index']]
        data['prediction'] = [str(data_all[x]['prediction']) for x in data['index']]

        dump(data, result_file)
        for i in range(world_size):
            os.remove(tmpl.format(i))
        if os.path.exists(prev_file):
            os.remove(prev_file)
    if world_size > 1:
        dist.barrier()
    
    
def evaluate(image_root, dataset, model, tokenizer, image_processor, work_dir, template):
    rank, world_size = get_rank_and_world_size()
    for c, data_dict in dataset.items():
        # 先将原有json文件转换为padnas的dataframe，便于后续处理
        rows = []
        for index, (key, val) in enumerate(data_dict.items()):
            row = [index, val['filename'], val['negative_caption'], val['caption']]
            rows.append(row)
        df = pd.DataFrame(rows, columns=['index','image_path', 'negative_caption', 'caption'])
        infer_data_job(df, model, image_root, tokenizer, image_processor, c, work_dir, template)
        
        if world_size > 1:
            dist.barrier()

def main(args):
    # Initialize distributed environment
    rank, world_size = get_rank_and_world_size()
    if world_size > 1:
        local_rank = os.environ.get('LOCAL_RANK', 0)
        torch.cuda.set_device(int(local_rank))
        dist.init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=3600))
        
    model_path = args.model

    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path,args.model_base, model_name,torch_dtype="bfloat16", device_map='cpu')
    model.cuda()
    model.eval()
    
    # conv_template="qwen_1_5"
    global conv_template
    conv_template = "plain"

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
    
    if 'checkpoint' in model_path:
        model_paths = model_path.split('/')
        output_path = args.output + '/' + model_paths[-2] + '/' + model_paths[-1]
    else:
        output_path = args.output + '/' + model_name
    os.makedirs(output_path, exist_ok=True)
    
    
    evaluate(args.coco_image_root, dataset, model, tokenizer, image_processor, output_path, args.template)
    
    rank,_= get_rank_and_world_size()
    if rank == 0:
        metrics = {}
        for c, _ in dataset.items():
            data = load(osp.join(output_path, f'{c}.xlsx'))
            correct = sum([int(x) for x in data['prediction']])
            total = len(data)
            metrics[c] = correct / total
        metrics['average'] = sum(metrics.values()) / len(metrics)
        print(metrics)
        print(f"Dump results to: {os.path.join(output_path, f'results.json')}")
        json.dump(metrics, open(os.path.join(output_path, f'results.json'), 'w'), indent=4)
    
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default= "/root/checkpoints/projectors/llavanext-clip-vit-large-patch14-336-openai-Qwen2-0.5B-tune_mlp-template-qwen_1_5-lr1e-3-bs64")
    parser.add_argument('--output', type=str, default='./outputs', help="Directory to where results are saved")
    parser.add_argument('--coco_image_root', type=str, default='/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/public/pywang/cktan/data/coco/val2017')
    parser.add_argument('--model_base', type=str, default=None)
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--template', type=str, default='plain', help="Template for inference")

    args = parser.parse_args()
    
    main(args)