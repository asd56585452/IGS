import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
from pytorch3d import io
from pytorch3d.renderer import (
    PointsRasterizationSettings, 
    PointsRasterizer)
from pytorch3d.structures import Pointclouds
from pytorch3d.utils.camera_conversions import cameras_from_opencv_projection
import cv2
from igs.utils.graphics_utils import getWorld2View2, focal2fov, fov2focal

from igs.utils.typing import *

ValidScale = Union[Tuple[float, float], Num[Tensor, "2 D"]]

def scale_tensor(
    dat: Num[Tensor, "... D"], inp_scale: ValidScale, tgt_scale: ValidScale
):
    if inp_scale is None:
        inp_scale = (0, 1)
    if tgt_scale is None:
        tgt_scale = (0, 1)
    if isinstance(tgt_scale, Tensor):
        assert dat.shape[-1] == tgt_scale.shape[-1]
    dat = (dat - inp_scale[0]) / (inp_scale[1] - inp_scale[0])
    dat = dat * (tgt_scale[1] - tgt_scale[0]) + tgt_scale[0]
    return dat

def scale_tensor_batch(
    dat: Num[Tensor, "... D"], inp_scale: ValidScale, tgt_scale: ValidScale
):
    if inp_scale is None:
        inp_scale = (0, 1)
    if tgt_scale is None:
        tgt_scale = (0, 1)
    if isinstance(tgt_scale, Tensor):
        assert dat.shape[-1] == tgt_scale.shape[-1]
    dat = (dat - inp_scale[:,0:1]) / (inp_scale[:,1:2] - inp_scale[:,0:1])
    dat = dat * (tgt_scale[1] - tgt_scale[0]) + tgt_scale[0]
    return dat
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


def get_activation(name) -> Callable:
    if name is None:
        return lambda x: x
    name = name.lower()
    if name == "none":
        return lambda x: x
    elif name == "lin2srgb":
        return lambda x: torch.where(
            x > 0.0031308,
            torch.pow(torch.clamp(x, min=0.0031308), 1.0 / 2.4) * 1.055 - 0.055,
            12.92 * x,
        ).clamp(0.0, 1.0)
    elif name == "exp":
        return lambda x: torch.exp(x)
    elif name == "shifted_exp":
        return lambda x: torch.exp(x - 1.0)
    elif name == "trunc_exp":
        return trunc_exp
    elif name == "shifted_trunc_exp":
        return lambda x: trunc_exp(x - 1.0)
    elif name == "sigmoid":
        return lambda x: torch.sigmoid(x)
    elif name == "tanh":
        return lambda x: torch.tanh(x)
    elif name == "shifted_softplus":
        return lambda x: F.softplus(x - 1.0)
    elif name == "scale_-11_01":
        return lambda x: x * 0.5 + 0.5
    else:
        try:
            return getattr(F, name)
        except AttributeError:
            raise ValueError(f"Unknown activation function: {name}")

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


def get_rays(
    directions: Float[Tensor, "... 3"],
    c2w: Float[Tensor, "... 4 4"],
    keepdim=False,
    noise_scale=0.0,
) -> Tuple[Float[Tensor, "... 3"], Float[Tensor, "... 3"]]:
    # Rotate ray directions from camera coordinate to the world coordinate
    assert directions.shape[-1] == 3

    if directions.ndim == 2:  # (N_rays, 3)
        if c2w.ndim == 2:  # (4, 4)
            c2w = c2w[None, :, :]
        assert c2w.ndim == 3  # (N_rays, 4, 4) or (1, 4, 4)
        rays_d = (directions[:, None, :] * c2w[:, :3, :3]).sum(-1)  # (N_rays, 3)
        rays_o = c2w[:, :3, 3].expand(rays_d.shape)
    elif directions.ndim == 3:  # (H, W, 3)
        assert c2w.ndim in [2, 3]
        if c2w.ndim == 2:  # (4, 4)
            rays_d = (directions[:, :, None, :] * c2w[None, None, :3, :3]).sum(
                -1
            )  # (H, W, 3)
            rays_o = c2w[None, None, :3, 3].expand(rays_d.shape)
        elif c2w.ndim == 3:  # (B, 4, 4)
            rays_d = (directions[None, :, :, None, :] * c2w[:, None, None, :3, :3]).sum(
                -1
            )  # (B, H, W, 3)
            rays_o = c2w[:, None, None, :3, 3].expand(rays_d.shape)
    elif directions.ndim == 4:  # (B, H, W, 3)
        assert c2w.ndim == 3  # (B, 4, 4)
        rays_d = (directions[:, :, :, None, :] * c2w[:, None, None, :3, :3]).sum(
            -1
        )  # (B, H, W, 3)
        rays_o = c2w[:, None, None, :3, 3].expand(rays_d.shape)

    # add camera noise to avoid grid-like artifect
    # https://github.com/ashawkey/stable-dreamfusion/blob/49c3d4fa01d68a4f027755acf94e1ff6020458cc/nerf/utils.py#L373
    if noise_scale > 0:
        rays_o = rays_o + torch.randn(3, device=rays_o.device) * noise_scale
        rays_d = rays_d + torch.randn(3, device=rays_d.device) * noise_scale

    rays_d = F.normalize(rays_d, dim=-1)
    if not keepdim:
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

    return rays_o, rays_d


def get_projection_matrix(
    fovy: Union[float, Float[Tensor, "B"]], aspect_wh: float, near: float, far: float
) -> Float[Tensor, "*B 4 4"]:
    if isinstance(fovy, float):
        proj_mtx = torch.zeros(4, 4, dtype=torch.float32)
        proj_mtx[0, 0] = 1.0 / (math.tan(fovy / 2.0) * aspect_wh)
        proj_mtx[1, 1] = -1.0 / math.tan(
            fovy / 2.0
        )  # add a negative sign here as the y axis is flipped in nvdiffrast output
        proj_mtx[2, 2] = -(far + near) / (far - near)
        proj_mtx[2, 3] = -2.0 * far * near / (far - near)
        proj_mtx[3, 2] = -1.0
    else:
        batch_size = fovy.shape[0]
        proj_mtx = torch.zeros(batch_size, 4, 4, dtype=torch.float32)
        proj_mtx[:, 0, 0] = 1.0 / (torch.tan(fovy / 2.0) * aspect_wh)
        proj_mtx[:, 1, 1] = -1.0 / torch.tan(
            fovy / 2.0
        )  # add a negative sign here as the y axis is flipped in nvdiffrast output
        proj_mtx[:, 2, 2] = -(far + near) / (far - near)
        proj_mtx[:, 2, 3] = -2.0 * far * near / (far - near)
        proj_mtx[:, 3, 2] = -1.0
    return proj_mtx


def get_mvp_matrix(
    c2w: Float[Tensor, "*B 4 4"], proj_mtx: Float[Tensor, "*B 4 4"]
) -> Float[Tensor, "*B 4 4"]:
    # calculate w2c from c2w: R' = Rt, t' = -Rt * t
    # mathematically equivalent to (c2w)^-1
    if c2w.ndim == 2:
        assert proj_mtx.ndim == 2
        w2c: Float[Tensor, "4 4"] = torch.zeros(4, 4).to(c2w)
        w2c[:3, :3] = c2w[:3, :3].permute(1, 0)
        w2c[:3, 3:] = -c2w[:3, :3].permute(1, 0) @ c2w[:3, 3:]
        w2c[3, 3] = 1.0
    else:
        w2c: Float[Tensor, "B 4 4"] = torch.zeros(c2w.shape[0], 4, 4).to(c2w)
        w2c[:, :3, :3] = c2w[:, :3, :3].permute(0, 2, 1)
        w2c[:, :3, 3:] = -c2w[:, :3, :3].permute(0, 2, 1) @ c2w[:, :3, 3:]
        w2c[:, 3, 3] = 1.0
    # calculate mvp matrix by proj_mtx @ w2c (mv_mtx)
    mvp_mtx = proj_mtx @ w2c
    return mvp_mtx

def get_intrinsic_from_fov(fov, H, W, bs=-1):
    focal_length = 0.5 * H / np.tan(0.5 * fov)
    intrinsic = np.identity(3, dtype=np.float32)
    intrinsic[0, 0] = focal_length
    intrinsic[1, 1] = focal_length
    intrinsic[0, 2] = W / 2.0
    intrinsic[1, 2] = H / 2.0

    if bs > 0:
        intrinsic = intrinsic[None].repeat(bs, axis=0)

    return torch.from_numpy(intrinsic)

def points_projection(points: Float[Tensor, "B Np 3"],
                    c2ws: Float[Tensor, "B 4 4"],
                    intrinsics: Float[Tensor, "B 3 3"],
                    local_features: Float[Tensor, "B C H W"],
                    # Rasterization settings
                    raster_point_radius: float = 0.0075,  # point size
                    raster_points_per_pixel: int = 1,  # a single point per pixel, for now
                    bin_size: int = 0):
    B, C, H, W = local_features.shape
    device = local_features.device
    raster_settings = PointsRasterizationSettings(
            image_size=(H, W),
            radius=raster_point_radius,
            points_per_pixel=raster_points_per_pixel,
            bin_size=bin_size,
        )
    Np = points.shape[1]
    R = raster_settings.points_per_pixel

    w2cs = torch.inverse(c2ws)
    image_size = torch.as_tensor([H, W]).view(1, 2).expand(w2cs.shape[0], -1).to(device)
    cameras = cameras_from_opencv_projection(w2cs[:, :3, :3], w2cs[:, :3, 3], intrinsics, image_size)

    rasterize = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    fragments = rasterize(Pointclouds(points))
    fragments_idx: Tensor = fragments.idx.long()
    visible_pixels = (fragments_idx > -1)  # (B, H, W, R)
    points_to_visible_pixels = fragments_idx[visible_pixels]

    # Reshape local features to (B, H, W, R, C)
    local_features = local_features.permute(0, 2, 3, 1).unsqueeze(-2).expand(-1, -1, -1, R, -1)  # (B, H, W, R, C)

    # Get local features corresponding to visible points
    local_features_proj = torch.zeros(B * Np, C, device=device)
    local_features_proj[points_to_visible_pixels] = local_features[visible_pixels]
    local_features_proj = local_features_proj.reshape(B, Np, C)

    return local_features_proj

def compute_distance_transform(mask: torch.Tensor):
    image_size = mask.shape[-1]
    # print(mask,mask.shape)
    # for m in mask.squeeze(1).detach().cpu().numpy().astype(np.uint8):
    #     print(1-m,m.shape,type(m))
    #     m = m[:,np.newaxis]
    #     cv2.distanceTransform(src = m, distanceType=cv2.DIST_L2, maskSize=cv2.DIST_MASK_3)
    distance_transform = torch.stack([
        torch.from_numpy(cv2.distanceTransform(
            (1 - m), distanceType=cv2.DIST_L2, maskSize=cv2.DIST_MASK_3
        ) / (image_size / 2))
        for m in mask.squeeze(1).detach().cpu().numpy().astype(np.uint8)
    ]).unsqueeze(1).clip(0, 1).to(mask.device)
    return distance_transform



def project_imagefeatures_to_3d(depth, FOV, cam_c2w_input):
    '''
    motion_feature:[B,C,H,W]
    depth:[B,H,W] 
    initri:[B,3,3] FOV[B,2]
    c2w:[B,4,4] #opencv/colmap/3dgs 的相机坐标系

    output:[B,3,HxW],得到每个像素在3D空间下的坐标
    '''
    B, H, W = depth.shape

    # 将深度图和图像特征展平
    depth_flat = depth.view(B, H * W)
    # motion_feature_flat = motion_feature.view(B, C, H * W)
    
    # 创建像素坐标网格
    y_coords, x_coords = torch.meshgrid(torch.arange(H), torch.arange(W))
    y_coords = y_coords.view(1, H, W).expand(B, H, W).to(depth.device)
    x_coords = x_coords.view(1, H, W).expand(B, H, W).to(depth.device)
    
    # 将像素坐标展平

    y_coords_flat = y_coords.contiguous().view(B, H * W)
    x_coords_flat = x_coords.contiguous().view(B, H * W)
    
    # 将像素坐标转换为相机坐标系中的坐标
    # cam_initri_input = cam_initri_input.view(B, 3, 3)

    FovX , FovY = FOV[:,0], FOV[:,1]
    fx = fov2focal(FovX, W)
    fy = fov2focal(FovY, H)

    # fx = cam_initri_input[:, 0, 0].view(B, 1)
    # fy = cam_initri_input[:, 1, 1].view(B, 1)
    # cx = cam_initri_input[:, 0, 2].view(B, 1)
    # cy = cam_initri_input[:, 1, 2].view(B, 1)
    
    x_cam = (x_coords_flat - W/2) * depth_flat / fx.unsqueeze(1)
    y_cam = (y_coords_flat - H/2) * depth_flat / fy.unsqueeze(1)
    z_cam = depth_flat
    # 将相机坐标系中的坐标转换为世界坐标系中的坐标
    ones = torch.ones_like(x_cam)
    points_cam = torch.stack((x_cam, y_cam, z_cam, ones), dim=2)
    
    # pcd = o3d.geometry.PointCloud()
    # points = points_cam[0,:,:3].detach().cpu().numpy()
    # print(points.shape)
    # print(points)
    # pcd.points = o3d.utility.Vector3dVector(points)
    # o3d.visualization.draw_geometries([pcd])

    # 相机外参矩阵
    cam_c2w_input = cam_c2w_input.view(B, 4, 4)
    
    # 计算世界坐标系中的坐标
    points_world = torch.bmm(cam_c2w_input, points_cam.transpose(1, 2)).transpose(1, 2)
    
    # 提取世界坐标系中的坐标
    points_world = points_world[:, :, :3]
    return points_world


# def get_triplane(feat_3d_pos, motion_feature, triplane_size):
    

def test(motion_feature, cam_initri_input, cam_c2w_input, depth):
    pcd_total = o3d.geometry.PointCloud()
    print(depth.shape)
    for i in range(motion_feature.shape[0]):
        points = project_imagefeatures_to_3d(motion_feature[i:i+1], cam_initri_input[i:i+1], cam_c2w_input[i:i+1], depth[i:i+1])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.visualization.draw_geometries([pcd])

        pcd_total += pcd
    
    o3d.visualization.draw_geometries([pcd_total])
def json_to_camera(idx=0):
    with open("../datasets/n3d/cameras.json") as f:
        data = json.load(f)[idx]
    w2c = np.zeros((4, 4))
    w2c[:3,:3] = np.array(data["rotation"])
    w2c[:3,3] = np.array(data["position"])
    w2c[3,3] = 1
    # c2w = np.linalg.inv(w2c)
    c2w = w2c

    inrinstic = np.zeros((3, 3))
    inrinstic[0,0] = data["fx"]/2
    inrinstic[1,1] = data["fy"]/2
    inrinstic[0,2] = data["width"]/4
    inrinstic[1,2] = data["height"]/4
    print(data["width"],data["height"])

    
    return inrinstic,c2w