import os
import re
import json
import numpy as np
from argparse import ArgumentParser


# exp_name = "3dgs-mcube"
# exp_name = "3dgs-poisson"
# exp_name = "3dgs-poisson"
parser = ArgumentParser(description="Training script parameters") 

parser.add_argument("-d", "--exp_name", type=str, default=None)
args = parser.parse_args()

# exp_name = "3dgs-neus-best-no-mask-HP10-no-reg-bary-field"
exp_name = args.exp_name

output_dir = 'output/NeRF_Syn'  # 文件夹路径
# output_dir = 'output/DTU'  # 文件夹路径
psnr_list = []
ssim_list = []
lpips_list = []
name_list = []

# 遍历文件夹和文件
# for root, dirs, files in os.listdir(output_dir):
dirs = os.listdir(output_dir)
dirs.sort()
for dir_name in dirs:
    eval_path = os.path.join(output_dir, dir_name, exp_name, "eval", "eval.txt")
    with open(eval_path, 'r') as f:
        eval_content = f.read()
        # 使用正则表达式匹配三个指标值
        psnr = re.findall(r'psnr: ([\d.]+)', eval_content)[0]
        ssim = re.findall(r'ssim: ([\d.]+)', eval_content)[0]
        lpips = re.findall(r'lpips: ([\d.]+)', eval_content)[0]
        name_list.append(dir_name)
        psnr_list.append(float(psnr))
        ssim_list.append(float(ssim))
        lpips_list.append(float(lpips))

# 将指标值存到json文件中
result_dict = {
    'name': name_list,
    'avg_psnr': np.mean(psnr_list),
    'avg_ssim': np.mean(ssim_list),
    'avg_lpips': np.mean(lpips_list),
    'psnr': psnr_list,
    'ssim': ssim_list,
    'lpips': lpips_list
}
with open('result_'+exp_name+'.json', 'w') as f:
    json.dump(result_dict, f, indent=4)