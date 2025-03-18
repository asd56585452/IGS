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
parser = ArgumentParser("subsample 2x")

parser.add_argument("--root_path", "-r", required=True, type=str)
parser.add_argument("--target_size", "-s", required=True, type=int, nargs=2, help="Target size for resizing (width height)")

args = parser.parse_args()
root_path = args.root_path
target_size = tuple(args.target_size)  # Convert to tuple (height, width)

def sample(images):
    images = F.interpolate(images, size=(512, 512), mode='bilinear', align_corners=False)

# for i in tqdm(range(300)):
def sub(i):
    print(i)
    image_path = os.path.join(root_path, f"colmap_{i}", "images")
    # image_path = os.path.join(root_path, f"colmap_{i}", "3dgs_rade","train", "ours_10000_compress","gt")

    res_path = os.path.join(root_path, f"colmap_{i}", "images_r2")
    os.makedirs(res_path, exist_ok=True)
    images = []
    image_names = []
    for name in os.listdir(image_path):
        if name.endswith('.png'):
            image_name = os.path.join(image_path, name)
            print(image_name)
            image_pil = Image.open(image_name)
            image_pil=image_pil.resize(target_size)
            image_data = torch.from_numpy(np.array(image_pil)/255.0).permute(2,0,1).to(torch.float)
            torchvision.utils.save_image(image_data, os.path.join(res_path, name))


res = []
p = mp.Pool(30)
for path in tqdm(range(0,300)):
    # print(i)
    res.append(p.apply_async(sub, args=(path,)))

p.close()
p.join()
print(res)