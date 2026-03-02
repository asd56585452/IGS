# 将图像subsample到 512x512,cpu 并行
from einops import rearrange
import os
import torchvision
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
import imageio
from tqdm import tqdm
from argparse import ArgumentParser

import multiprocessing as mp
parser = ArgumentParser("subsample to 512")

parser.add_argument("--root_path", "-r", required=True, type=str)

args = parser.parse_args()
root_path = args.root_path

def sub(i):
    print("process: ", i)
    image_path = os.path.join(root_path, f"colmap_{i}", "images")
    res_path = os.path.join(root_path, f"colmap_{i}", "images_512")
    os.makedirs(res_path, exist_ok=True)
    images = []
    image_names = []
    for name in os.listdir(image_path):
        if name.endswith('.png') or name.endswith('.jpg'):
            image_name = os.path.join(image_path, name)
            image_data = torch.from_numpy(np.array(Image.open(image_name))/255.0).permute(2,0,1).to(torch.float)
            # Process individually to handle variable sizes
            image_data = image_data.unsqueeze(0) # Add batch dim for interpolate
            image_data = F.interpolate(image_data, size=(512, 512), mode='bilinear', align_corners=False)
            image_data = image_data.squeeze(0) # Remove batch dim
            torchvision.utils.save_image(image_data, os.path.join(res_path, name))

res = []
p = mp.Pool(5)
for path in tqdm(range(0,300)):
    res.append(p.apply_async(sub, args=(path,)))

p.close()
p.join()
print(res)