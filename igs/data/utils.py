import os
import cv2
import random
import json
from dataclasses import dataclass, field
from einops import rearrange
import numpy as np
from PIL import Image
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from igs.utils.typing import *
from igs.utils.config import parse_structured
from torchvision import transforms as T
from igs.utils.ops import get_intrinsic_from_fov
from igs.utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from igs.utils.general_utils import getNerfppNorm

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
def _parse_scene_list_single(scene_list_path: str):
    if scene_list_path.endswith(".json"):
        with open(scene_list_path) as f:
            all_scenes = json.loads(f.read())
    elif scene_list_path.endswith(".txt"):
        with open(scene_list_path) as f:
            all_scenes = [p.strip() for p in f.readlines()]
    else:
        all_scenes = [scene_list_path]

    return all_scenes


def _parse_scene_list(scene_list_path: Union[str, List[str]]):
    all_scenes = []
    if isinstance(scene_list_path, str):
        scene_list_path = [scene_list_path]
    for scene_list_path_ in scene_list_path:
        all_scenes += _parse_scene_list_single(scene_list_path_)
    return all_scenes



def build_rays(c2ws, ixts, H, W, scale=1.0):

    H, W = int(H*scale), int(W*scale)
    ixts[:,:2] *= scale

    rays_o = c2ws[:,:3, 3][:,None,None]
    X, Y = np.meshgrid(np.arange(W), np.arange(H))
    XYZ = np.concatenate((X[:, :, None] + 0.5, Y[:, :, None] + 0.5, np.ones_like(X[:, :, None])), axis=-1)
    i2ws = np.linalg.inv(ixts).transpose(0,2,1) @ c2ws[:,:3, :3].transpose(0,2,1)
    XYZ = np.stack([(XYZ @ i2w) for i2w in i2ws])
    rays_o = rays_o.repeat(H, axis=1)
    rays_o = rays_o.repeat(W, axis=2)
    rays = np.concatenate((rays_o, XYZ), axis=-1)
    return rays.astype(np.float32)

def get_ray_directions(
    H: int,
    W: int,
    focal: Union[float, Tuple[float, float]],
    principal: Optional[Tuple[float, float]] = None,
    use_pixel_centers: bool = True,
) -> Float[Tensor, "H W 3"]:
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        H, W, focal, principal, use_pixel_centers: image height, width, focal length, principal point and whether use pixel centers
    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    pixel_center = 0.5 if use_pixel_centers else 0

    if isinstance(focal, float):
        fx, fy = focal, focal
        cx, cy = W / 2, H / 2
    else:
        fx, fy = focal
        assert principal is not None
        cx, cy = principal

    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32) + pixel_center,
        torch.arange(H, dtype=torch.float32) + pixel_center,
        indexing="xy",
    )

    directions: Float[Tensor, "H W 3"] = torch.stack(
        [(i - cx) / fx, -(j - cy) / fy, -torch.ones_like(i)], -1
    )

    return directions
