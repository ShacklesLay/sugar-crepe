import matplotlib.pyplot as plt
import sys
sys.path.append('/home/image_data/cktan/reps/server_tools')
from scripts import load, dump
import os
import numpy as np

dir_path = './outputs'

def get_data_plot(output_dir):
    output_dir = os.path.join(dir_path, output_dir)
    ck_files = os.listdir(output_dir)
    ck_files_paths = [os.path.join(output_dir, f) for f in ck_files]
    ck_files_paths = sorted(ck_files_paths, key=lambda x: int(x.split('-')[-1]))
    
    data_plot = {}
    
    for path in ck_files_paths:
        data = load(os.path.join(path, 'results.json'))
        if 'average' not in data:
            data['average'] = sum(data.values()) / len(data)
        for k, v in data.items():
            if k not in data_plot:
                data_plot[k] = []
            data_plot[k].append(v)
    return data_plot

def insert_epoch0(data_plot, data, key_name):
    for k, v in data.items():
        data_plot[key_name][k].insert(0, v)
    return data_plot

# 重排data_plot的嵌套顺序，将中间层放到最前面
def reorder(data_plot):
    new_data_plot = {}
    for k, v in data_plot.items():
        for key_name, values in v.items():
            if key_name not in new_data_plot:
                new_data_plot[key_name] = {}
            new_data_plot[key_name][k] = values
    return new_data_plot
        
data_plot = {}

data_plot['baseline_epoch5'] = get_data_plot('llavanext-clip-vit-large-patch14-336-openai-Qwen2-0.5B-tune_mlp-select_last_layer-midtune_vt-blip558k-epochs_5')
data_plot['cmov3_epoch3'] = get_data_plot('llavanext-clip-vit-large-patch14-336-openai-Qwen2-0.5B-cmov3-tune_mlp_lastlayer-cmov3-midtune_vt-blip558k-epochs_3')
data_plot['cmov2_epoch3'] = get_data_plot('llavanext-clip-vit-large-patch14-336-openai-Qwen2-0.5B-cmov2-tune_mlp_lastlayer_lr1e-3_bs64-cmov2-midtune_vt-blip558k-epochs_3')
data_plot['cmov2_tokenloss_epoch3'] = get_data_plot('llavanext-clip-vit-large-patch14-336-openai-Qwen2-0.5B-cmov2_token_loss-tune_mlp_lastlayer-cmov2token-midtune_vt-blip558k-epochs_3')

data_plot = insert_epoch0(data_plot, load('./outputs/projectors/llavanext-clip-vit-large-patch14-336-openai-Qwen2-0.5B-cmov3-tune_mlp_lastlayer/results.json'), 'cmov3_epoch3')
data_plot = insert_epoch0(data_plot, load('./outputs/projectors/llavanext-clip-vit-large-patch14-336-openai-Qwen2-0.5B-tune_mlp-select_last_layer/results.json'), 'baseline_epoch5')
data_plot = insert_epoch0(data_plot, load('./outputs/projectors/llavanext-clip-vit-large-patch14-336-openai-Qwen2-0.5B-cmov2-tune_mlp_lastlayer_lr1e-3_bs64/results.json'), 'cmov2_epoch3')
data_plot = insert_epoch0(data_plot, load('./outputs/projectors/llavanext-clip-vit-large-patch14-336-openai-Qwen2-0.5B-cmov2_token_loss-tune_mlp_lastlayer/results.json'), 'cmov2_tokenloss_epoch3')

# print(data_plot)
data_plot = reorder(data_plot)

baseline_data = load("/home/image_data/cktan/reps/sugar-crepe/outputs/projectors/llavanext-clip-vit-large-patch14-336-openai-Qwen2-0.5B-tune_mlp-select_last_layer/results.json")
baseline_data['average'] = sum(baseline_data.values()) / len(baseline_data)

llm_data = load("/home/image_data/cktan/reps/sugar-crepe/outputs/Qwen2-0.5B/results.json")
llm_data['average'] = sum(llm_data.values()) / len(llm_data)

clip_data = load("/home/image_data/cktan/reps/sugar-crepe/outputs/clip-vit-large-patch14-336-openai/results.json")
clip_data['average'] = sum(clip_data.values()) / len(clip_data)

output_dir = dir_path+'/cmo'
os.makedirs(output_dir, exist_ok=True)
# 创建 8 张图，每张图对应一个列表
for i, (key, line) in enumerate(data_plot.items()):
    plt.figure(i)  # 每次绘制新的一张图
    
    baseline = baseline_data[key]
    llm = llm_data[key]
    clip = clip_data[key]
    # e0 = cmov2_mlp[key]
    # blip558k = blip558k_data[key]
    
    for line_name, values in line.items():
        x_values = np.arange(0, len(values))  # 创建横坐标
        plt.plot(x_values,values, label=line_name)  # 绘制折线图
    
    plt.xticks(np.arange(0, len(values), 1))
    plt.axhline(y=baseline, color='r', linestyle='--', label='Baseline')
    plt.axhline(y=llm, color='b', linestyle='--', label='LLM')
    plt.axhline(y=clip, color='g', linestyle='--', label='CLIP')
    
    plt.title(f'Line Plot for {key}')  # 设置图标题
    plt.xlabel('Epoch')  # 设置 x 轴标签
    plt.ylabel('Value')  # 设置 y 轴标签
    plt.legend()  # 显示图例
    plt.grid(True)  # 显示网格

    save_path = os.path.join(output_dir, f'{key}_plot.png')  # 使用键作为文件名的一部分
    plt.savefig(save_path, dpi=300)  # 保存图像，dpi 控制分辨率（300 为高质量）
    
    # 清理当前图，以便绘制下一张图
    plt.close()  # 关闭当前图像，防止图像堆积在内存中