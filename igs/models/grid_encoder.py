import torch
import torch.nn as nn
import torch.nn.functional as F
from igs.utils.ops import project_imagefeatures_to_3d, scale_tensor_batch, select_points_bbox, perspective_projection, points_projection
# from torch_scatter import scatter_mean, scatter_max
from igs.utils.base import BaseModule
from dataclasses import dataclass, field
from einops import rearrange
import igs
from icecream import ic
import numpy as np
from igs.utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from torch_cluster import fps, knn



class GridEncoder(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        plane_size:int = 256
        in_channels:int = 128
        out_channels:int = 256
        num_views:int = 0

        aggregate:str = "mean"
        proj_type:str = "PointsRasterizer"
        unet_cls: str = ""
        unet: dict = field(default_factory=dict)
        combine_type: str = "mask"

        transformer_cls: str =""
        transformer: dict = field(default_factory=dict) 

        camera_embedder_cls: str =""
        camera_embedder: dict = field(default_factory=dict)

        grid_type: str = "grid"
        use_gs_emb: bool = False
        res_cat: bool = False
    cfg = Config

    def configure(self) -> None:
        super().configure()

        if self.cfg.combine_type == "attention":
            self.transformer = igs.find(self.cfg.transformer_cls)(self.cfg.transformer)
            self.proj_out = nn.Linear(256,128)

            self.camera_embedder = igs.find(self.cfg.camera_embedder_cls)(**self.cfg.camera_embedder) #这个是继承的nn.Module,并不是basemodule
        self.conv = igs.find(self.cfg.unet_cls)(self.cfg.unet)


    def forward(self, 
        motion_feature,
        gs_emb,
        anchor_points,
        FOV,
        c2w_input,
        bounding_box,
        mask_list = None,
        
        **kwargs
    #  FOV, cam_c2w_input, depth, bounding_box, batch_size, points, gs_emb, anchor_points, mask_list =None
     ):
 
        batch_size = c2w_input.shape[0]

        view_num = c2w_input.shape[1]
        FOV = FOV.unsqueeze(1).repeat(1, view_num,1).reshape(-1,2)
        c2w_input = rearrange(c2w_input, "B V H W -> (B V) H W")


        if self.cfg.proj_type == "perspective_projection":
            #the one used in paper
            W, H = motion_feature.shape[-2:]
            fx , fy = fov2focal(FOV[0,0], W), fov2focal(FOV[0,1], H) 
            intrinsic = np.identity(3, dtype=np.float32)
            intrinsic[0, 0] = fx
            intrinsic[1, 1] = fy
            intrinsic[0, 2] = W / 2.0
            intrinsic[1, 2] = H / 2.0
            intrinsics = torch.from_numpy(intrinsic[None].repeat(FOV.shape[0], axis=0)).to(c2w_input)

            points_proj = anchor_points.unsqueeze(1).repeat_interleave(motion_feature.shape[0]//batch_size, 1)
            points_proj = rearrange(points_proj, "B V N D -> (B V) N D")
            proj_feats = perspective_projection(points_proj, c2w_input, intrinsics, motion_feature.to(torch.float))
            proj_feats = rearrange(proj_feats , "(B V) N D -> B V N D", B = batch_size)
            motion_grids = proj_feats.mean(dim=1)





        if self.cfg.grid_type == "irgrid":
            if self.cfg.use_gs_emb:

                emb_grids = gs_emb
                feature_grid = torch.cat([motion_grids, emb_grids], dim=-1)
            else:
                feature_grid = motion_grids
            feature_grid = self.conv(feature_grid.permute(0,2,1)) # use transformer block to get global information
            feature_grid = feature_grid.permute(0,2,1)

            if self.cfg.res_cat:
                feature_grid = torch.cat([feature_grid, motion_grids], dim=-1)

        return feature_grid





