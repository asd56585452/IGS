import torch
import torch.nn as nn
from dataclasses import dataclass, field
from einops import rearrange
import os
from torch.utils.data import DataLoader
from einops import rearrange

from igs.utils.saving import SaverMixin
from igs.utils.config import parse_structured
from igs.utils.misc import load_module_weights
from igs.utils.typing import *
import igs
import torch.nn.functional as F
from igs.utils.base import BaseModule
from kiui.lpips import LPIPS
from icecream import ic
import numpy as np
from igs.models.gs import get_mask_fpsample, get_mask_no_fpsample
from collections import defaultdict

class IGS(torch.nn.Module, SaverMixin):
    @dataclass
    class Config:
        weights: Optional[str] = None
        weights_ignore_modules: Optional[List[str]] = None

        transformer_cls: str = ""
        transformer: dict = field(default_factory=dict)

        renderer_cls: str = ""
        renderer: dict = field(default_factory=dict)

        renderer_emb_cls: str = ""
        renderer_emb: dict = field(default_factory=dict)

        backbone_cls: str = ""
        backbone: dict = field(default_factory=dict)

        triplane_encoder_cls: str = ""
        triplane_encoder: dict = field(default_factory=dict)

        combine_net_cls:str = ""
        combine_net: dict = field(default_factory=dict)

        temporal_net_cls:str = ""
        temporal_net: dict = field(default_factory=dict)

        use_point_generator: bool = False

        local_ray: bool = True
        up_sample: bool = False

        frame_num: int = 1
        use_gs_emb: bool =  False

        temporal: bool = False
        compute_once:  Optional[List[str]] = None
        SV: bool = False

        use_condition3d: bool = True
        fine_tune_backbone: bool = True
    cfg: Config

    def load_weights(self, weights: str, ignore_modules: Optional[List[str]] = None):
        state_dict = load_module_weights(
            weights, ignore_modules=ignore_modules, map_location="cpu"
        )
        self.load_state_dict(state_dict, strict=False)

    def __init__(self, cfg):
        super().__init__()
        self.cfg = parse_structured(self.Config, cfg)
        self._save_dir: Optional[str] = None
        self.backbone = igs.find(self.cfg.backbone_cls)(self.cfg.backbone)
        self.backbone.requires_grad_(False)
        self.backbone.eval()

        if self.cfg.use_gs_emb:
            self.render_emb = igs.find(self.cfg.renderer_emb_cls)(self.cfg.renderer_emb)

        ic(self.cfg.combine_net_cls, self.cfg.combine_net)

        # self.combine_net = igs.find(self.cfg.combine_net_cls)(self.cfg.combine_net)
        if self.cfg.fine_tune_backbone:
            self.transformer = igs.find(self.cfg.transformer_cls)(self.cfg.transformer)

        self.triplane_encoder = igs.find(self.cfg.triplane_encoder_cls)(self.cfg.triplane_encoder)
        self.render = igs.find(self.cfg.renderer_cls)(self.cfg.renderer)

        if self.cfg.use_condition3d:
            if self.cfg.local_ray:
                self.ModLN = ModLN(128, 4, eps=1e-6)
            else:
                self.ModLN = ModLN(128, 33, eps=1e-6)

        if self.cfg.up_sample:
            self.upsample = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.stream_eval = False
        self.stream_eval_batch = False
        self.ab_anchor_driven = False
        if self.cfg.temporal:
            self.learnable_history = nn.Parameter(torch.zeros(8096, 256))
            self.temporal_net = igs.find(self.cfg.temporal_net_cls)(self.cfg.temporal_net)
    def state_dict(self, **kwargs):
        # remove lpips_loss
        state_dict = super().state_dict(**kwargs)
        for k in list(state_dict.keys()):
            if 'backbone' in k:
                del state_dict[k]
        return state_dict


    def _forward_v3(self, batch: Dict[str, Any], first_frame = True) -> Dict[str, Any]:
        #the forward of AGM-Net 

        B,V, C,H,W = batch["cur_images_input"].shape 

        # get 2D motion feature
        cur_images_input = batch["cur_images_input"].reshape((-1, C, H, W))
        next_images_input = batch["next_images_input"].reshape((-1, C, H, W))

        img_feature_0, img_feature_1 = self.backbone(cur_images_input, next_images_input) #[(B V) N C H/8 W/8]
        
        if self.cfg.fine_tune_backbone:
            motion_feature = self.transformer(img_feature_0, img_feature_1,
                                                attn_type="swin",
                                                attn_num_splits=2) #[(B V) C H/8 W/8]
        else:
            motion_feature = img_feature_0

        if self.cfg.up_sample:
            motion_feature = F.interpolate(motion_feature,scale_factor=2, mode='bilinear', align_corners=False)
            motion_feature = self.upsample(motion_feature)

        resolution = motion_feature.shape[-2:]

        if self.stream_eval_batch:
            #pre compute
            anchor_points, mask_list, weights, neighbor, anchor_idx = batch["stream_eval_batch"]
            embedding_3dgs = None
        else:
            if self.cfg.use_gs_emb:
                # compute the embedding of gaussian, not used in the final version
                embedding_3dgs, anchor_points, mask_list, weights, neighbor, anchor_idx = self.render_emb(resolution, **batch)
                if "v_gs" in batch and batch["v_gs"]!=None:

                    temp_batch = batch.copy()
                    temp_batch["gs"] = batch["v_gs"]
                    del temp_batch["v_gs"]
                    embedding_3dgs,anchor_points,_,_,_,_ = self.render_emb(resolution, **temp_batch)
            else:
                if self.ab_anchor_driven:
                    # ablation no anchor driven
                    anchor_points, mask_list, weights, neighbor, anchor_idx = get_mask_no_fpsample(batch["gs"], batch["bounding_box"])
                else:
                    # the method used in the paper
                    anchor_points, mask_list, weights, neighbor, anchor_idx = get_mask_fpsample(batch["gs"], batch["bounding_box"])
                embedding_3dgs = None




        if self.cfg.use_condition3d:
            # inject the depth and camera information
            feature_H, feature_W = motion_feature.shape[-2], motion_feature.shape[-1]
            motion_feature = self.condition3D(motion_feature, batch["rays"] if not  self.cfg.local_ray else batch["local_rays"], batch["depth"])

        triplane = self.triplane_encoder( motion_feature, embedding_3dgs, anchor_points, mask_list = mask_list, **batch) #[B 3 C h w]
        



        res = self.render( triplane, mask_list, anchor_points=anchor_points, weights=weights, neighbor=neighbor, **batch)
        if first_frame:
            pre_compute_states = {"masks_precompute":mask_list, "anchor_points": anchor_points, "weights": weights, "neighbor":neighbor, "embedding_3dgs":embedding_3dgs, "fps_idx_precompute": anchor_idx}

            res.update({"pre_compute_states": pre_compute_states})

        res.update({"motion_feature": triplane.clone()})
        return {**res}  

   

    def condition3D(self, motion_feature, rays, depth):
        view_num = depth.shape[1]


        depth = rearrange(depth, "B V H W -> (B V) 1 H W")
        resolution = motion_feature.shape[-2:]

        if self.cfg.local_ray:
            ray = rays.unsqueeze(1).repeat_interleave(view_num, dim=1)
            ray = rearrange(ray, "B V H W C -> (B V) H W C")
        else:

            ray = ray_to_plucker(rays)
            ray = torch.cat((rsh_cart_3(ray[...,:3]),rsh_cart_3(ray[...,3:6])),dim=-1)
            ray = rearrange(ray, "B V H W C -> (B V) H W C")

        depth = F.interpolate(depth, size=resolution, mode='bilinear', align_corners=False).squeeze(dim=1)

        ray_condition = torch.cat([ray, depth.unsqueeze(-1)], dim=-1)

        motion_feature = self.ModLN(rearrange(motion_feature, "B C H W -> B H W C"), ray_condition)

        
        motion_feature = rearrange(motion_feature, "B H W C -> B C H W ")

        return motion_feature

    def set_stream_eval(self):
        self.stream_eval = True
        self.pre_compute_states = None

    def reset_pre_compute_states(self):
        self.pre_compute_states = None

    def forward(self, batch):

        # out = self._forward_v3(batch)
        if self.stream_eval_batch:
            anchor_points, mask_list, weights, neighbor, anchor_idx = get_mask_fpsample(batch["gs"], batch["bounding_box"])
            row, col, batch_x, batch_y = neighbor

            batch_size = batch["bounding_box"].shape[0]
            anchor_size = anchor_points.shape[1]
            point_size = batch_y.shape[0]

            anchor_points = anchor_points.repeat(batch_size,1,1)
            weights = weights.repeat(batch_size,1,1)
            mask_list = mask_list * batch_size
            anchor_idx = anchor_idx * batch_size

            new_batch_x = [batch_x]
            new_batch_y = [batch_y]
            new_row = [row]
            new_col = [col]

            for i in range(1, batch_size):
                new_batch_x.append(batch_x+i)
                new_batch_y.append(batch_y+i)
                new_row.append(row+point_size*i)
                new_col.append(col+anchor_size*i)
            new_batch_x = torch.cat(new_batch_x,dim=0)
            new_batch_y = torch.cat(new_batch_y,dim=0)
            new_row = torch.cat(new_row,dim=0)
            new_col = torch.cat(new_col,dim=0)
            neighbor = (new_row, new_col,new_batch_x, new_batch_y)
            batch["stream_eval_batch"] = (anchor_points, mask_list, weights, neighbor, anchor_idx)
            batch["gs"] = batch["gs"]*batch_size
            out = self._forward_v3(batch)
            return out
        else:

            out = self._forward_v3(batch)
        return out

class ModLN(nn.Module):
    """
    Modulation with adaLN.
    
    References:
    DiT: https://github.com/facebookresearch/DiT/blob/main/models.py#L101
    """
    def __init__(self, inner_dim: int, mod_dim: int, eps: float):
        super().__init__()
        self.norm = nn.LayerNorm(inner_dim, eps=eps)
        hidden_dim = 128
        self.mlp = nn.Sequential(
            nn.Linear(mod_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, inner_dim * 2),
        )

    @staticmethod
    def modulate(x, shift, scale):
        # x: [N, L, D]
        # shift, scale: [N, D]
        return x * (1 + scale) + shift

    def forward(self, x, cond):
        shift, scale = self.mlp(cond).chunk(2, dim=-1)  # [N, D]
        return self.modulate(self.norm(x), shift, scale)  # [N, L, D]

def ray_to_plucker(rays):
    origin, direction = rays[...,:3], rays[...,3:6]
    # Normalize the direction vector to ensure it's a unit vector
    direction = F.normalize(direction, p=2.0, dim=-1)
    
    # Calculate the moment vector (M = O x D)
    moment = torch.cross(origin, direction, dim=-1)
    
    # Plucker coordinates are L (direction) and M (moment)
    return torch.cat((direction, moment),dim=-1)

def rsh_cart_3(xyz: torch.Tensor):
    """Computes all real spherical harmonics up to degree 3.

    This is an autogenerated method. See
    https://github.com/cheind/torch-spherical-harmonics
    for more information.

    Params:
        xyz: (N,...,3) tensor of points on the unit sphere

    Returns:
        rsh: (N,...,16) real spherical harmonics
            projections of input. Ynm is found at index
            `n*(n+1) + m`, with `0 <= n <= degree` and
            `-n <= m <= n`.
    """
    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]

    x2 = x**2
    y2 = y**2
    z2 = z**2
    xy = x * y
    xz = x * z
    yz = y * z

    return torch.stack(
        [
            xyz.new_tensor(0.282094791773878).expand(xyz.shape[:-1]),
            -0.48860251190292 * y,
            0.48860251190292 * z,
            -0.48860251190292 * x,
            1.09254843059208 * xy,
            -1.09254843059208 * yz,
            0.94617469575756 * z2 - 0.31539156525252,
            -1.09254843059208 * xz,
            0.54627421529604 * x2 - 0.54627421529604 * y2,
            -0.590043589926644 * y * (3.0 * x2 - y2),
            2.89061144264055 * xy * z,
            0.304697199642977 * y * (1.5 - 7.5 * z2),
            1.24392110863372 * z * (1.5 * z2 - 0.5) - 0.497568443453487 * z,
            0.304697199642977 * x * (1.5 - 7.5 * z2),
            1.44530572132028 * z * (x2 - y2),
            -0.590043589926644 * x * (x2 - 3.0 * y2),
        ],
        -1,
    )
