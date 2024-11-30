import matplotlib.pyplot as plt
import sys
sys.path.append('/home/image_data/cktan/reps/server_tools')
from scripts import load
import os
import numpy as np

dir_path = './outputs'
files = os.listdir(dir_path)

files = [f for f in files if f.startswith('llavanext-clip-vit-large-patch14-336-openai-Qwen2-0.5B-tune_mlp-blip3kale558k-select_last_layer-midtune_vt_mlp-blip3kale3M-select_last_layer')]
files = sorted(files, key=lambda x: int(x.split('-')[-1]))

data_plot = {}
for f in files:
    path = os.path.join(dir_path, f)
    data = load(os.path.join(path, 'results.json'))
    if 'average' not in data:
        data['average'] = sum(data.values()) / len(data)
    for k, v in data.items():
        if k not in data_plot:
            data_plot[k] = []
        data_plot[k].append(v)


baseline_data = load("/home/image_data/cktan/reps/sugar-crepe/outputs/llavanext-clip-vit-large-patch14-336-openai-Qwen2-0.5B-tune_mlp-select_last_layer/results.json")
baseline_data['average'] = sum(baseline_data.values()) / len(baseline_data)

llm_data = load("/home/image_data/cktan/reps/sugar-crepe/outputs/Qwen2-0.5B/results.json")
llm_data['average'] = sum(llm_data.values()) / len(llm_data)

clip_data = load("/home/image_data/cktan/reps/sugar-crepe/outputs/clip-vit-large-patch14-336-openai/results.json")
clip_data['average'] = sum(clip_data.values()) / len(clip_data)

output_dir = dir_path+'/kale558k_kale3M_plots'
os.makedirs(output_dir, exist_ok=True)
# 创建 8 张图，每张图对应一个列表
for i, (key, values) in enumerate(data_plot.items()):
    plt.figure(i)  # 每次绘制新的一张图
    x_values = np.arange(1, len(values) + 1)  # 创建横坐标 1 到 10
    plt.plot(x_values,values, label='kale558k+kale3M')  # 绘制折线图
    
    baseline = baseline_data[key]
    llm = llm_data[key]
    clip = clip_data[key]
    
    plt.axhline(y=baseline, color='r', linestyle='--', label='Baseline')
    plt.axhline(y=llm, color='b', linestyle='--', label='LLM')
    plt.axhline(y=clip, color='g', linestyle='--', label='CLIP')
    
    plt.title(f'Line Plot for {key}')  # 设置图标题
    plt.xlabel('Million')  # 设置 x 轴标签
    plt.ylabel('Value')  # 设置 y 轴标签
    plt.legend()  # 显示图例
    plt.grid(True)  # 显示网格

    save_path = os.path.join(output_dir, f'{key}_plot.png')  # 使用键作为文件名的一部分
    plt.savefig(save_path, dpi=300)  # 保存图像，dpi 控制分辨率（300 为高质量）
    
    # 清理当前图，以便绘制下一张图
    plt.close()  # 关闭当前图像，防止图像堆积在内存中