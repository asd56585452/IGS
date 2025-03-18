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

import torch
import sys
from datetime import datetime
import numpy as np
import random
import cv2

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))


def hook_fn(grad, name):
    # print(f"Gradient of the tensor '{name}': {grad}")
    print("grad:",name, grad.max(), grad.min(), torch.isnan(grad).any())
    return grad

def visualize_flow(flow_pr):
    # flow = padder.unpad(flow_pr[0]).detach().cpu().numpy()
    if len(flow_pr.shape) ==4:
        flow_pr = flow_pr[0]
    flow = flow_pr.detach().cpu().numpy()
    flow_x = flow[0, :, :]
    flow_y = flow[1, :, :]
    
    # 计算光流的大小和方向
    magnitude, angle = cv2.cartToPolar(flow_x, flow_y)
    
    # 归一化幅度到 [0, 255]
    magnitude_normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    magnitude_normalized = magnitude_normalized.astype(np.uint8)
    
    # 将幅度图转换为彩色图像
    flow_image = cv2.applyColorMap(magnitude_normalized, cv2.COLORMAP_JET)
    return flow_image

def getNerfppNorm(cam_centers):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal
    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

@torch.compile()
def quaternion_multiply(a, b):
    """
    Multiply two sets of quaternions.
    
    Parameters:
    a (Tensor): A tensor containing N quaternions, shape = [N, 4]
    b (Tensor): A tensor containing N quaternions, shape = [N, 4]
    
    Returns:
    Tensor: A tensor containing the product of the input quaternions, shape = [N, 4]
    """
    assert a.shape[0] == b.shape[0]
    a_norm=torch.nn.functional.normalize(a)
    b_norm=torch.nn.functional.normalize(b)
    # print("nor a", a[0])
    # print("nor b", b[0])
    w1, x1, y1, z1 = a_norm[:, 0], a_norm[:, 1], a_norm[:, 2], a_norm[:, 3]
    w2, x2, y2, z2 = b_norm[:, 0], b_norm[:, 1], b_norm[:, 2], b_norm[:, 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2

    return torch.stack([w, x, y, z], dim=1)

def quaternion_to_rotation_vector(q):
    """ 将四元数转换为旋转向量 """
    q = q.to(torch.float32)
    q = torch.nn.functional.normalize(q)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    theta = 2 * torch.acos(w)
    mask = theta != 0  # 创建一个掩码，用于处理 theta 为 0 的情况
    r = torch.zeros_like(q[:, 1:])
    # print(r.dtype, q.dtype, theta.dtype)
    with torch.autocast(device_type=q.device.type, dtype=torch.float32):
        r[mask] = (theta[mask] / torch.sin(theta[mask] / 2)).unsqueeze(1) * q[mask, 1:]
    # print(q, theta, r)
    return r

def rotation_vector_to_quaternion(r):
    """ 将旋转向量转换为四元数 """
    theta = torch.norm(r, dim=1)
    mask = theta != 0  # 创建一个掩码，用于处理 theta 为 0 的情况
    q = torch.zeros((r.shape[0], 4), dtype=r.dtype, device=r.device)
    q[:, 0] = torch.cos(theta / 2)
    q[mask, 1:] = (torch.sin(theta[mask] / 2) / theta[mask]).unsqueeze(1) * r[mask]
    return q