#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
from dataclasses import dataclass, field
from igs.models.networks import MLP
from einops import rearrange
import fpsample

import torch
import torch.nn.functional as F
import numpy as np
from igs.utils.general_utils import get_expon_lr_func, build_rotation
from torch import nn
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
import os
from igs.utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
import open3d as o3d
from igs.utils.typing import *
from igs.utils.base import BaseModule
from collections import defaultdict
from igs.utils.ops import scale_tensor_batch, scale_tensor, select_points_bbox, position_embedding

from igs.utils.sh_utils import RGB2SH
# from simple_knn._C import distCUDA2
from igs.utils.graphics_utils import BasicPointCloud, fov2focal
from igs.utils.general_utils import inverse_sigmoid, get_expon_lr_func, strip_symmetric, build_scaling_rotation, build_rotation, hook_fn, quaternion_multiply, quaternion_to_rotation_vector, rotation_vector_to_quaternion
# from mmcv.ops import knn
import math
from diff_gaussian_rasterization_rade import GaussianRasterizationSettings as GaussianRasterizationSettings_rade, GaussianRasterizer as GaussianRasterizer_rade
from diff_gaussian_rasterization_rade_clamp import GaussianRasterizationSettings as GaussianRasterizationSettings_rade_clamp, GaussianRasterizer as GaussianRasterizer_rade_clamp

from torch_cluster import fps, knn
from torch_scatter import scatter_mean, scatter_max
from icecream import ic


inverse_sigmoid = lambda x: np.log(x / (1 - x))

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def intrinsic_to_fov(intrinsic, w, h):
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    fov_x = 2 * torch.arctan2(w, 2 * fx)
    fov_y = 2 * torch.arctan2(h, 2 * fy)
    return fov_x, fov_y


class Camera:
    def __init__(self, w2c, FoVx, FoVy, resolution, trans=np.array([0.0, 0.0, 0.0]), scale=1.0) -> None:
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.height = resolution[0]
        self.width = resolution[1]
        self.world_view_transform = w2c.transpose(0, 1)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).to(w2c.device)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    @staticmethod
    def from_c2w(c2w, FOV, resolution):
        w2c = torch.inverse(c2w)

        FoVx, FoVy = FOV[0], FOV[1]
        return Camera(w2c=w2c, FoVx=FoVx, FoVy=FoVy, resolution=resolution)
        
def get_embedder(multires, i=1):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : i,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2'] #L-1
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)
    
# def sample_point(N):
def build_rot(forward):
    #以3dgs里面的坐标系，即up轴是向下的
    # local2world
    #forward :[N,3]
    up = torch.tensor((0,-1,0),dtype=torch.float,device="cuda").unsqueeze(0).repeat(forward.shape[0],1)
    right = torch.cross(up,F.normalize(forward,dim=-1))
    up = torch.cross(forward, right)
    return torch.stack((right,up,forward),dim=-1)

def sphere2xyz(radius,sita,phi):
    #sita [0,2pi] phi[0,pi]
    x = radius*torch.cos(sita)*torch.sin(phi)
    y = radius*torch.sin(sita)*torch.sin(phi)
    z = radius*torch.cos(phi)
    point = torch.stack((x,y,z),dim=-1) #[...,3]
    return point

class _TruncExp(Function):  # pylint: disable=abstract-method
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):  # pylint: disable=arguments-differ
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=15))


trunc_exp = _TruncExp.apply
class GaussianModel(NamedTuple):
    xyz: Tensor
    opacity: Tensor
    rotation: Tensor
    scaling: Tensor
    shs: Tensor

    resi_xyz: Tensor = None
    resi_rotation: Tensor = None
    mask: Tensor = None #mask of the points in bbox

    def setup_functions(self):
        # def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
        #     L = build_scaling_rotation(scaling_modifier * scaling, rotation)
        #     actual_covariance = L @ L.transpose(1, 2)
        #     symm = strip_symmetric(actual_covariance)
        #     return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        # self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize



    @property
    def get_scaling(self):
        exp_scaling = torch.exp(self.scaling)

        return exp_scaling

    @property
    def get_rotation(self):
        return F.normalize(self.rotation)
    
    @property
    def get_xyz(self):
        return self.xyz

    @property
    def get_bounding_box(self):
        xyz = self.get_xyz
        bounding_min = xyz.amin(dim=0, keepdim=True)
        bounding_max = xyz.amax(dim=0, keepdim=True)
        bounding_box = torch.cat([bounding_min, bounding_max], dim=0) #[2,3]
        return bounding_box
    


    @property
    def get_features(self):
        return self.shs


    @property
    def get_opacity(self):
        # return self.opacity
        return torch.sigmoid(self.opacity )



    @property
    def get_deform_scaling(self):
        return self.scaling_activation(self._scaling + self.res_scaling)

    @property
    def get_deform_rotation(self):
        return self.rotation_activation(self._rotation + self.res_rot)
    
    @property
    def get_deform_xyz(self):
        return self._xyz + self.res_xyz

    @property
    def get_deform_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1) + self.res_sh

    @property
    def get_deform_opacity(self):
        return self.opacity_activation(self._opacity + self.res_opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1



   
    def construct_list_of_attributes(self, sh_degree=3):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(3):
            l.append('f_dc_{}'.format(i))
        # resr_num = 3*(sh_degree+1)**2 - 3
        # for i in range(9):
        for i in range(45):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self.scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self.rotation.shape[1]):
            l.append('rot_{}'.format(i))
        # l.append('d2f')
        return l


 

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))


        mask = torch.ones_like(self.get_opacity).squeeze().to(torch.bool) #mask nothing

        xyz = self.xyz[mask].detach().cpu().numpy()
        normals = np.zeros_like(xyz)

        f_dc = self.get_features[mask][:,0:1].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self.get_features[mask][:,1:].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        
        opacities = self.opacity[mask].detach().cpu().numpy()
        scale = self.scaling[mask].detach().cpu().numpy()
        rotation = self.rotation[mask].detach().cpu().numpy()


        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)


        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)


    
    
    def deform(self, res_feat, mask):
        '''
        res_feat:dict
        '''
        if mask ==None:
            mask = torch.range(self.xyz.shape[0])
        

        final={}
        final["xyz"] = self.xyz.clone()
        final["opacity"] = self.opacity.clone()
        final["rotation"] = self.rotation.clone()
        final["scaling"] = self.scaling.clone()
        final["shs"] = self.shs.clone()

        for key in res_feat:
            if key == "rotation":
                final[key][mask] = quaternion_multiply(final[key][mask], res_feat["rotation"].clone())
                final["resi_rotation"] = res_feat["rotation"].clone()

            elif key == "shs":
                final[key][mask] = res_feat["shs"].clone().reshape(-1,16,3) + final[key][mask]
            else:
                final[key][mask] = res_feat[key].clone()  + final[key][mask]

        final["mask"] = mask
        final["resi_xyz"] = res_feat["xyz"].clone()
        gs = GaussianModel(**final)
        return gs

    def lbs_deform(self, res_feat, mask):
        '''
        res_feat:dict
        '''
        if mask ==None:
            mask = torch.range(self.xyz.shape[0])
        

        final={}
        final["xyz"] = self.xyz.clone()
        final["opacity"] = self.opacity.clone()
        final["rotation"] = self.rotation.clone()
        final["scaling"] = self.scaling.clone()
        final["shs"] = self.shs.clone()

        final["resi_xyz"] = res_feat["xyz"] - final["xyz"][mask]
        final["xyz"][mask] = res_feat["xyz"].clone().to(final["xyz"])
        final["rotation"][mask] = res_feat["rot"].clone()
        final["resi_rotation"] = res_feat["drot"].clone()
        final["mask"] = mask
        gs = GaussianModel(**final)
        return gs

def load_ply( path, max_sh_degree=3):
    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)

    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
    

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    assert len(extra_f_names)==3*(max_sh_degree + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])


    filter_3D = np.asarray(plydata.elements[0]["filter_3D"])[..., np.newaxis]
    filter_3D = torch.tensor(filter_3D, dtype=torch.float)

    xyz = torch.tensor(xyz, dtype=torch.float)



    features_dc = torch.tensor(features_dc, dtype=torch.float).transpose(1, 2).contiguous()
    features_rest = torch.tensor(features_extra, dtype=torch.float).transpose(1, 2).contiguous()
    
    shs = torch.cat((features_dc, features_rest), dim=1)
    
    opacity = torch.tensor(opacities, dtype=torch.float)
    scaling = torch.tensor(scales, dtype=torch.float)
    rotation = torch.tensor(rots, dtype=torch.float)

    scaling_act, opacity_act =  get_scaling_n_opacity_with_3D_filter( scaling, opacity, filter_3D)
    

    opacity = inverse_sigmoid(opacity_act)
    # opacity = opacity_act
    scaling = torch.log(scaling_act)
    gs = GaussianModel(xyz, opacity, rotation, scaling, shs)


    return gs
def load_from_radegs(gsmodel):
    if gsmodel.mask != None:
        select_index = torch.arange(gsmodel._xyz.shape[0], device="cuda")
        select_index[gsmodel.mask] = torch.arange(gsmodel.xyz_dynamic.shape[0], device="cuda") + gsmodel.outbox_xyz.shape[0]
        select_index[~gsmodel.index_bool] = torch.arange(gsmodel.outbox_xyz.shape[0], device="cuda")
        xyz = torch.index_select(gsmodel._xyz, dim=0, index = select_index)
        shs = torch.index_select(gsmodel._shs, dim=0, index=select_index)
        opacity = torch.index_select(gsmodel._opacity, dim=0, index=select_index)
        rotation = torch.index_select(gsmodel._rotation, dim=0, index=select_index)
        scaling = torch.index_select(gsmodel._scaling, dim=0, index=select_index)
        gs = GaussianModel(xyz, opacity, rotation, scaling, shs)
        # gs = GaussianModel(gsmodel._xyz, gsmodel._opacity, gsmodel._rotation, gsmodel._scaling, gsmodel._shs)

    else:
        gs = GaussianModel(gsmodel.get_xyz, gsmodel._opacity, gsmodel.get_rotation, gsmodel._scaling, gsmodel._shs)
    return gs

def get_scaling_n_opacity_with_3D_filter(scaling, opacity, filter_3D):
    opacity = torch.sigmoid(opacity)
    scales = torch.exp(scaling)
    scales_square = torch.square(scales)
    det1 = scales_square.prod(dim=1)
    scales_after_square = scales_square + torch.square(filter_3D) 
    det2 = scales_after_square.prod(dim=1) 
    coef = torch.sqrt(det1 / det2)

    scales = torch.sqrt(scales_after_square)
    return scales, opacity * coef[..., None]

   
def xavier_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class GS3DRenderer(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        mlp_network_config: Optional[dict] = None
        # gs_out: dict = field(default_factory=dict)
        sh_degree: int = 3
        scaling_modifier: float = 1.0
        random_background: bool = False
        radius: float = 1.0
        feature_reduction: str = "concat"
        projection_feature_dim: int = 773
        background_color: Tuple[float, float, float] = field(
            default_factory=lambda: (1.0, 1.0, 1.0)
        )
        in_channels: int = 124
        feature_channels: dict = field(default_factory=dict)
        xyz_offset: bool = True
        restrict_offset: bool = False
        use_rgb: bool = False
        clip_scaling: Optional[float] = None
        init_scaling: float = -5.0
        init_density: float = 0.1

        xyz_scale : float = 0.01
        ret_rgb: bool = True

        feature_mode: str ="irgrid" # or  triplane
        render_flow: bool = False
        flow_height: int = 1024
        flow_width: int = 1352

        interpolate_first: bool = True
        lbs: bool = False
        neighbor_size: int = 8
    cfg: Config

    def configure(self, *args, **kwargs) -> None:

        mlp_in=self.cfg.mlp_network_config.n_neurons
        if self.cfg.mlp_network_config is not None:
            self.mlp_net = MLP(mlp_in, self.cfg.in_channels, **self.cfg.mlp_network_config)
        else:
            self.cfg.gs_out.in_channels = mlp_in

        self.out_layers = nn.ModuleList()

        for key, out_ch in self.cfg.feature_channels.items():
            if key == "shs" and self.cfg.use_rgb:
                out_ch = 3
            layer = nn.Linear(self.cfg.in_channels, out_ch)


            if key =="rotation":
                nn.init.constant_(layer.weight, 0)
                layer.bias.data = torch.tensor([1, 1e-2, 1e-2, 1e-2], dtype=torch.float32)
            else:
                nn.init.constant_(layer.weight, 0)
                nn.init.constant_(layer.bias, 0)


            self.out_layers.append(layer)

    def forward_gs(self, x, p,neighbors):
        if self.cfg.mlp_network_config is not None:
            x = self.mlp_net(x)
        return self.gs_net(x, p,neighbors)

    def forward_single_view(self,
        gs: GaussianModel,
        viewpoint_camera: Camera,
        background_color: Optional[Float[Tensor, "3"]],
        ret_mask: bool = True,
        ret_rgb: bool = True,
        # ret_flow: bool =Fasle,
        original_gs: GaussianModel = None
        ):
        # with torch.autocast(device_type=self.device.type, dtype=torch.float32):
            ret = {}
            
            bg_color = background_color
            # Set up rasterization configuration
            tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
            tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
            if ret_rgb:
                screenspace_points = torch.zeros_like(gs.xyz, dtype=gs.xyz.dtype, requires_grad=True, device=self.device) + 0
                try:
                    screenspace_points.retain_grad()
                except:
                    pass
                


                raster_settings = GaussianRasterizationSettings_rade_clamp(
                    image_height=int(viewpoint_camera.height),
                    image_width=int(viewpoint_camera.width),
                    tanfovx=tanfovx,
                    tanfovy=tanfovy,
                    bg=bg_color,
                    scale_modifier=self.cfg.scaling_modifier,
                    viewmatrix=viewpoint_camera.world_view_transform,
                    projmatrix=viewpoint_camera.full_proj_transform.float(),
                    sh_degree=self.cfg.sh_degree,
                    campos=viewpoint_camera.camera_center,
                    prefiltered=False,
                    debug=False,

                    #rade-GS的默认参数
                    kernel_size = 0.0,
                    require_coord = True,
                    require_depth = True,

                )
                rasterizer = GaussianRasterizer_rade_clamp(raster_settings=raster_settings)

                means3D = gs.get_xyz

                means2D = screenspace_points.contiguous().float()
                opacity = gs.get_opacity
                # .contiguous().float()

                # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
                # scaling / rotation by the rasterizer.
                scales = None
                rotations = None
                cov3D_precomp = None
                # scales = gs.get_scaling
                scales = gs.get_scaling #这里暂时这样写
                # exp_sca_hook = scales.register_hook(lambda grad: hook_fn(grad, "exp_sca_hook"))

                # .contiguous().float()
                rotations = gs.get_rotation
                # .contiguous().float()

                # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
                # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
                shs = None
                colors_precomp = None
                if self.cfg.use_rgb:
                    colors_precomp = gs.shs.squeeze(1).contiguous().float()
                else:
                    shs = gs.get_features
                    # .contiguous().float()

                rendered_image, radii,rendered_expected_coord, rendered_median_coord, rendered_expected_depth, rendered_median_depth, rendered_alpha, rendered_normal = rasterizer(
                    means3D = means3D,
                    means2D = means2D,
                    shs = shs,
                    colors_precomp = colors_precomp,
                    opacities = opacity,
                    scales = scales,
                    rotations = rotations,
                    cov3D_precomp = cov3D_precomp)


                ret = {
                    "images_pred": rendered_image,#.permute(1, 2, 0),
                    "bg_color": bg_color,
                    "depth_pred": rendered_expected_depth
                }
            
            if original_gs != None:
                with torch.autocast(device_type=original_gs.xyz.device.type, dtype=torch.float32):

                    raster_settings = GaussianRasterizationSettings_rade_clamp(
                        image_height=int(self.cfg.flow_height),
                        image_width=int(self.cfg.flow_width),
                        tanfovx=tanfovx,
                        tanfovy=tanfovy,
                        bg=bg_color,
                        scale_modifier=self.cfg.scaling_modifier,
                        viewmatrix=viewpoint_camera.world_view_transform,
                        projmatrix=viewpoint_camera.full_proj_transform.float(),
                        sh_degree=self.cfg.sh_degree,
                        campos=viewpoint_camera.camera_center,
                        prefiltered=False,
                        debug=False,

                        #rade-GS的默认参数
                        kernel_size = 0.0,
                        require_coord = True,
                        require_depth = True,

                    )
                    rasterizer = GaussianRasterizer_rade_clamp(raster_settings=raster_settings)

                    #渲染flow
                    mask = gs.mask
                    means3D = original_gs.get_xyz[mask]

                    means2D = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device=self.device) + 0
                    opacity = original_gs.get_opacity[mask]
                    scales = original_gs.get_scaling[mask] #这里暂时这样写
                    rotations = original_gs.get_rotation[mask]

                    # resi_xyz = torch.cat([gs.resi_xyz, torch.ones((gs.resi_xyz.shape[0],1)).to(gs.resi_xyz)], dim=-1)
                    resi_xyz = gs.resi_xyz
                    flow_cam = resi_xyz@viewpoint_camera.world_view_transform[:3,:3]
                    fx, fy = fov2focal(viewpoint_camera.FoVx,self.cfg.flow_width), fov2focal(viewpoint_camera.FoVy, self.cfg.flow_height)
                    
                    flow_2d = resi_xyz.new_zeros((resi_xyz.shape[0], 3))
                    flow_2d[:,0] = flow_cam[:,0]*fx/(means3D[:,2]+1e-6)
                    flow_2d[:,1] = flow_cam[:,1]*fy/(means3D[:,2]+1e-6)
                    colors_precomp = flow_2d.to(torch.float)
                    shs = None
                    cov3D_precomp=None
                    rendered_image, radii,rendered_expected_coord, rendered_median_coord, rendered_expected_depth, rendered_median_depth, rendered_alpha, rendered_normal = rasterizer(
                        means3D = means3D,
                        means2D = means2D,
                        shs = shs,
                        colors_precomp = colors_precomp,
                        opacities = opacity,
                        scales = scales,
                        rotations = rotations,
                        cov3D_precomp = cov3D_precomp)
                ret.update({"flow_pred":rendered_image[:2], "flow_mask":rendered_alpha})


            return ret
    
   

    def interpolate_residual_feats(
        self,
        resi_feats,
        weights,
        neighbor,
        ):

        points_resi_feats = {}
        row, col, batch_x, batch_y = neighbor

        for key in resi_feats:
            v = resi_feats[key]
            if key == "xyz":
                v = rearrange(v, "B N D -> (B N) D")
                v = rearrange(v[col], "(N K) D -> N K D", K=8)

                point_v= torch.sum(v * weights, dim=1)
            elif key == "rotation":
                v = rearrange(v, "B N D -> (B N) D")
                v = v[col]
                v = F.normalize(v)
                v = rearrange(v, "(N K) D -> N K D", K=8)
                point_v = torch.sum(v * weights, dim=1)
            else:
                raise NotImplementedError

            unique_indices, counts = torch.unique(batch_y, return_counts=True)
            # split based on group
            grouped_attrs = torch.split(point_v, counts.tolist())
            points_resi_feats[key] = grouped_attrs

        per_gs_resi_feats = []

        for i in range(len(unique_indices)):
            per_gs_resi_feats.append({k:v[i] for k, v in points_resi_feats.items()})
        
        return per_gs_resi_feats


    def query_ir_grid(
        self,
        positions: Float[list, "[N D]"],
        anchors: Float[Tensor, "*B N D"],
        anchor_feats: Float[Tensor, "*B N D"],
        bbox = None,
        mask_list = None,
        weights = None,
        neighbor = None
    ) -> Dict[str, Tensor]:
        batched = anchors.ndim == 3


        if not batched:
            # no batch dimension
            grids = grids[None, ...]
            positions = [positions]
            if bbox is not None:
                bbox = bbox[None, ...]
        
        batchsize = anchors.shape[0]

        if weights != None and neighbor !=None:
            row, col, batch_x, batch_y = neighbor

        else:
            if bbox is not None:
                index = select_points_bbox(positions, bbox)
                points_inbbox = positions[index]
            else:
                points = []
                batch_y = []
                for idx in range(batchsize):
                    xyz = positions[idx][mask_list[idx]]
                    batch = torch.zeros(xyz.shape[0])+ idx
                    points.append(xyz)
                    batch_y.append(batch)
                points = torch.cat(points, dim=0)
                batch_y = torch.cat(batch_y, dim=0).to(points.device)
            
            anchor_flatten = rearrange(anchors, "B N D -> (B N) D")
            

            batch_x =  torch.arange(batchsize).to(anchor_flatten.device)
            batch_x = torch.repeat_interleave(batch_x, anchor_flatten.shape[0]//batchsize)

            row, col = knn(anchor_flatten, points, 8,batch_x, batch_y)

            dist = torch.linalg.vector_norm(anchor_flatten[col] - points[row], ord=2, dim=-1)


            weights = torch.softmax( -10*dist.view(-1,self.cfg.neighbor_size), dim=-1).unsqueeze(-1)
        # features = features[col].view(-1,8)
        features = rearrange(anchor_feats, "B N D -> (B N) D")

        features = rearrange(features[col], "(N K) D -> N K D", K=self.cfg.neighbor_size)

        features= torch.sum(features * weights, dim=1) # N 8 1* N 8 D  -> N 8 D sum ->N D
        
        unique_indices, counts = torch.unique(batch_y, return_counts=True)
        # split based on group
        grouped_attrs = torch.split(features, counts.tolist())
        return grouped_attrs


    def forward_single_batch(
        self,
        resi_features: Dict,
        gaussian: GaussianModel,
        c2ws: Float[Tensor, "Nv 4 4"],
        FOV,
        resolution,
        background_color: Optional[Float[Tensor, "3"]],
        mask
    ):
        if self.cfg.lbs:
            gs = gaussian.lbs_deform(resi_features, mask)
        else:
            gs = gaussian.deform(resi_features, mask)

        out_list = []
        for c2w in c2ws:
            out_list.append(self.forward_single_view(
                                gs, 
                                Camera.from_c2w(c2w, FOV, resolution),
                                background_color,
                                ret_rgb = self.cfg.ret_rgb,
                                original_gs = gs if self.cfg.render_flow else None
                            ))
        
        out = defaultdict(list)
        for out_ in out_list:
            for k, v in out_.items():
                out[k].append(v)

        out = {k: torch.stack(v, dim=0) for k, v in out.items()}
        out["3dgs"] = gs
        return out

    def decode_residual_feature(self, residual_feat, bbox=None, radius=None):
        if self.cfg.mlp_network_config is not None:
            residual_feat = self.mlp_net( residual_feat)

        ret = {}
        for k, layer in zip(self.cfg.feature_channels.keys(), self.out_layers):
            v = layer(residual_feat)


            ret[k] = v
 
        return ret

    def lbs_deform(self, anchor_residual_feat, anchor_points, gs_list, weights, neighbor, mask_list):
        row, col, batch_x, batch_y = neighbor
        rot = anchor_residual_feat["rotation"].reshape(-1,4)
        rot = F.normalize(rot)
        rot_matrix = build_rotation(rot)
        T = anchor_residual_feat["xyz"].reshape(-1,3)
        gs_points = []
        gs_rots = []
        for idx in range(len(gs_list)):
            gs_points.append(gs_list[idx].get_xyz[mask_list[idx]])
            gs_rots.append(gs_list[idx].get_rotation[mask_list[idx]])
        gs_points = torch.cat(gs_points, dim=0)
        gs_rots = torch.cat(gs_rots, dim=0)
        anchor_points = anchor_points.reshape(-1,3)

        gs_points = gs_points[row]

        anchor_points = anchor_points[col]
        rot_matrix = rot_matrix[col]
        T = T[col]
        new_xyz = torch.bmm(rot_matrix, (gs_points - anchor_points).unsqueeze(-1)).squeeze(-1) + T+ anchor_points #[(N*8) 3]
        new_xyz = torch.sum(new_xyz.reshape(-1,self.cfg.neighbor_size,3)*weights, dim=1)

        rot = rot[col]
        drot = torch.sum(rot.reshape(-1,self.cfg.neighbor_size,4)*weights, dim=1)
        new_rot = quaternion_multiply(gs_rots, drot)
        
        unique_indices, counts = torch.unique(batch_y, return_counts=True)
        # split
        grouped_rot = torch.split(new_rot, counts.tolist())
        grouped_xyz = torch.split(new_xyz, counts.tolist())
        grouped_drot = torch.split(drot, counts.tolist())

        per_gs_resi_feats = []
        for i in range(len(unique_indices)):

            per_gs_resi_feats.append({"rot": grouped_rot[i], "xyz":grouped_xyz[i], "drot": grouped_drot[i]})
        
        return per_gs_resi_feats
        
    def forward(self, 
        # gaussians: GaussianModel,#gaussians
        features: Float[Tensor, "B Np Cp"],#triplane feature
        mask_list,
        gs,
        FOV,
        c2w_output,
        resolution,
        bounding_box,
        background_color,
        radius,
        anchor_points: Optional[Float[Tensor, "B N C"]] = None,
        additional_features: Optional[Float[Tensor, "B C H W"]] = None,
        weights = None,
        neighbor = None,
        **kwargs):
        batch_size = len(gs)
        out_list = []

        points = []
        for idx in range(len(gs)):
            points.append(gs[idx].get_xyz)
        if self.cfg.interpolate_first:
            if self.cfg.feature_mode =="irgrid":
                gs_hidden_features_list = self.query_ir_grid(points,anchor_points, features, bbox=None, mask_list = mask_list, weights=weights, neighbor = neighbor)

        for b in range(batch_size):
            if self.cfg.interpolate_first:
                if self.cfg.feature_mode =="irgrid":
                    gs_hidden_features = gs_hidden_features_list[b]
                residual_feat = self.decode_residual_feature( gs_hidden_features, bounding_box[b], radius[b])


            out_list.append(self.forward_single_batch(
                residual_feat,
                gs[b],
                c2w_output[b],
                FOV[b],
                resolution[b],
                background_color[b] if background_color is not None else None,
                mask_list[b]))

        out = defaultdict(list)
        for out_ in out_list:
            for k, v in out_.items():
                out[k].append(v)
        for k, v in out.items():
            if isinstance(v[0], torch.Tensor):
                out[k] = torch.stack(v, dim=0)
            else:
                out[k] = v
        return out
        


def get_mask_fpsample(gs_list, bbox,  anchor_size=8192):
    masks = []
    anchor_points = []
    batchsize = len(gs_list)
    points = []
    batch_y = []
    anchor_idx = []


    for idx, gs in enumerate(gs_list):
        #serial
        xyz = gs.get_xyz
        index_inbbox = select_points_bbox(xyz, bbox[idx])[1]
        masks.append(index_inbbox)
    

        pc = gs.get_xyz[index_inbbox]
        kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(pc.detach().cpu().numpy(), anchor_size, h=5)
        anchor_points.append(pc[torch.from_numpy(kdline_fps_samples_idx.astype(np.int64))])


        batch = torch.zeros(pc.shape[0])+ idx
        points.append(pc)
        batch_y.append(batch)
        anchor_idx.append(kdline_fps_samples_idx.astype(int))

    points = torch.cat(points, dim=0)
    batch_y = torch.cat(batch_y, dim=0).to(points.device)


    anchor_points = torch.stack(anchor_points, dim=0)

    anchor_flatten = rearrange(anchor_points, "B N D -> (B N) D")
    

    batch_x =  torch.arange(batchsize).to(anchor_flatten.device)
    batch_x = torch.repeat_interleave(batch_x, anchor_flatten.shape[0]//batchsize)

    row, col = knn(anchor_flatten, points, 8,batch_x, batch_y)

    dist = torch.linalg.vector_norm(anchor_flatten[col] - points[row], ord=2, dim=-1)


    weights = torch.softmax( -10*dist.view(-1,8), dim=-1).unsqueeze(-1)

    return anchor_points, masks, weights, (row, col , batch_x, batch_y), anchor_idx

def get_mask_no_fpsample(gs_list, bbox):
    #ablation study, don't use anchor
    masks = []
    anchor_points = []
    batchsize = len(gs_list)
    points = []
    batch_y = []
    anchor_idx = []

    for idx, gs in enumerate(gs_list):
        xyz = gs.get_xyz
        index_inbbox = select_points_bbox(xyz, bbox[idx])[1]
        masks.append(index_inbbox)


        pc = gs.get_xyz[index_inbbox]
        anchor_points.append(pc)


        batch = torch.zeros(pc.shape[0])+ idx
        points.append(pc)
        batch_y.append(batch)
        anchor_idx.append(index_inbbox)

    points = torch.cat(points, dim=0)
    batch_y = torch.cat(batch_y, dim=0).to(points.device)


    anchor_points = torch.stack(anchor_points, dim=0)

    anchor_flatten = rearrange(anchor_points, "B N D -> (B N) D")
    

    batch_x =  torch.arange(batchsize).to(anchor_flatten.device)
    batch_x = torch.repeat_interleave(batch_x, anchor_flatten.shape[0]//batchsize)

    row = torch.arange(points.shape[0], device="cuda")
    col = torch.arange(points.shape[0],device="cuda")

    weights = torch.ones((points.shape[0],1,1), device="cuda")
    return anchor_points, masks, weights, (row, col , batch_x, batch_y), anchor_idx

