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
parser = ArgumentParser("subsample images")

parser.add_argument("--root_path", "-r", required=True, type=str)
parser.add_argument("--target_size", "-s", type=int, nargs=2, default=None, help="Target size for resizing (width height)")
parser.add_argument("--ratio", type=float, default=None, help="Downsample ratio (e.g., 2 for 2x smaller)")
parser.add_argument("--out_dir", type=str, default="images_r2", help="Output folder name in each colmap folder")

args = parser.parse_args()
root_path = args.root_path

if args.target_size is None and args.ratio is None:
    parser.error("Either --target_size or --ratio must be provided")

if args.target_size is not None:
    target_size = tuple(args.target_size)
else:
    target_size = None

def sample(images):
    images = F.interpolate(images, size=(512, 512), mode='bilinear', align_corners=False)

# for i in tqdm(range(300)):
def sub(i):
    print(i)
    image_path = os.path.join(root_path, f"colmap_{i}", "images")
    # image_path = os.path.join(root_path, f"colmap_{i}", "3dgs_rade","train", "ours_10000_compress","gt")

    res_path = os.path.join(root_path, f"colmap_{i}", args.out_dir)
    os.makedirs(res_path, exist_ok=True)
    images = []
    image_names = []
    for name in os.listdir(image_path):
        if name.endswith('.png'):
            image_name = os.path.join(image_path, name)
            print(image_name)
            image_pil = Image.open(image_name)
            
            if target_size is not None:
                new_size = target_size
            else:
                new_size = (int(image_pil.width / args.ratio), int(image_pil.height / args.ratio))
                
            resample_filter = getattr(Image, 'Resampling', Image).LANCZOS
            image_pil = image_pil.resize(new_size, resample_filter)
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