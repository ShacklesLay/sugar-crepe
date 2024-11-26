import argparse
import json
import os
import os.path as osp
from PIL import Image
from tqdm import tqdm
import copy
import pandas as pd
import datetime

import torch
import torch.distributed as dist

import sys
sys.path.append("/remote-home1/cktan/server_tools/")
from scripts import dump, load, get_rank_and_world_size
from larknotice import lark_sender

from transformers import CLIPModel, AutoTokenizer, CLIPImageProcessor

def get_ppl(text, model, tokenizer, verbose=False):
    inputs_ids = tokenizer(text, return_tensors="pt").input_ids.cuda()
    
    cont = model(inputs_ids, labels=inputs_ids, return_dict=True)
    return cont.loss.item()

def text_retrieval(pos_text, neg_text, image, model, tokenizer, image_processor):
    pos_text = tokenizer(pos_text, return_tensors="pt").to('cuda')
    pos_text_embedding = model.get_text_features(**pos_text)
    neg_text = tokenizer(neg_text, return_tensors="pt").to('cuda')
    neg_text_embedding = model.get_text_features(**neg_text)
    image = image_processor(image, return_tensors='pt').to('cuda')
    image_embedding = model.get_image_features(**image)
    pos_score = pos_text_embedding @ image_embedding.t()
    neg_score = neg_text_embedding @ image_embedding.t()
    return 1 if pos_score.item() > neg_score.item() else 0


def infer_data(model, tokenizer, image_processor, work_dir, dataset, out_file, dataset_name, image_root):
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
        prediction = text_retrieval(pos_text, neg_text, image, model, tokenizer, image_processor)
        torch.cuda.empty_cache()
        
        res[idx] = {
            'prediction': prediction
        }
        
        if i % 100 == 0:
            dump(res, out_file)
    
    res = {k: res[k] for k in data_indices}
    dump(res, out_file)
        

# A wrapper for infer_data, do the pre & post processing
def infer_data_job(dataset, model, image_root, tokenizer, image_processor, dataset_name, work_dir):
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
        model, tokenizer, image_processor, work_dir=work_dir, dataset=dataset, out_file=out_file, dataset_name=dataset_name, image_root=image_root)
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
        
        data['prediction'] = [str(data_all[x]['prediction']) for x in data['index']]

        dump(data, result_file)
        for i in range(world_size):
            os.remove(tmpl.format(i))
        if os.path.exists(prev_file):
            os.remove(prev_file)
    if world_size > 1:
        dist.barrier()
    
    
def evaluate(image_root, dataset, model, tokenizer, image_processor, work_dir):
    metrics = {}
    for c, data_dict in dataset.items():
        # 先将原有json文件转换为padnas的dataframe，便于后续处理
        rows = []
        for index, (key, val) in enumerate(data_dict.items()):
            row = [index, val['filename'], val['negative_caption'], val['caption']]
            rows.append(row)
        df = pd.DataFrame(rows, columns=['index','image_path', 'negative_caption', 'caption'])
        infer_data_job(df, model, image_root, tokenizer, image_processor, c, work_dir)

@lark_sender(webhook_url="https://open.feishu.cn/open-apis/bot/v2/hook/9824a4f2-07e2-40cc-ae32-74ded5a0db96")
def main(args, lark_task):
    # Initialize distributed environment
    rank, world_size = get_rank_and_world_size()
    if world_size > 1:
        local_rank = os.environ.get('LOCAL_RANK', 0)
        torch.cuda.set_device(int(local_rank))
        dist.init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=10800))
        
    model_path = args.model
    
    model = CLIPModel.from_pretrained(model_path, torch_dtype=torch.bfloat16).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    image_processor = CLIPImageProcessor.from_pretrained(model_path)

    model.eval()

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
    
    model_name = model_path.split('/')[-1]
    output_path = args.output + '/' + model_name
    os.makedirs(output_path, exist_ok=True)
    
    evaluate(args.coco_image_root, dataset, model, tokenizer, image_processor, output_path)
    
    rank,_= get_rank_and_world_size()
    if rank == 0:
        metrics = {}
        for c, _ in dataset.items():
            data = load(osp.join(output_path, f'{c}.xlsx'))
            correct = sum([int(x) for x in data['prediction']])
            total = len(data)
            metrics[c] = correct / total
        print(metrics)
        print(f"Dump results to: {os.path.join(output_path, f'results.json')}")
        json.dump(metrics, open(os.path.join(output_path, f'results.json'), 'w'), indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default= "/remote-home1/share/models/vision_encoder/clip-vit-large-patch14-336-openai")
    parser.add_argument('--output', type=str, default='./outputs', help="Directory to where results are saved")
    parser.add_argument('--coco_image_root', type=str, default='/remote-home1/share/data/COCO2017val/val2017')
    parser.add_argument('--model_base', type=str, default=None)
    parser.add_argument('--data_root', type=str, default='./data')

    args = parser.parse_args()
    
    lark_message = 'Evaluate Qwen2.5'
    main(args, lark_task=lark_message)