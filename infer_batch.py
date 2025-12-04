import time
from dataclasses import dataclass, field
import os
import random
import numpy as np
import torch

from accelerate import Accelerator, DistributedDataParallelKwargs
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from igs.IGS import IGS
import kiui
import torch.nn.functional as F
from igs.utils.typing import *
from igs.utils.config import parse_structured
import shutil, pathlib
from omegaconf import OmegaConf
from igs.utils.system_utils import mkdir_p,save_tensor_2_npy
from torch.utils.tensorboard import SummaryWriter
from einops import rearrange
import cv2
import math
from diff_gaussian_rasterization_rade import GaussianRasterizationSettings as GaussianRasterizationSettings_rade, GaussianRasterizer as GaussianRasterizer_rade

from igs.utils.loss_utils import l1_loss, ssim
import torch.nn as nn
import igs
from icecream import ic
from tqdm import tqdm
import json
import torchvision


from random import randint
from igs.models.gs import Camera
from igs.models.gs import load_from_radegs
from igs.utils.ops import scale_tensor_batch, scale_tensor, select_points_bbox, position_embedding
import time
def forward_single_view(
    gs,
    viewpoint_camera: Camera,
    background_color: Optional[Float[Tensor, "3"]],
    sh_degree = 3,
    # need_depth = False,
    ):
        ret = {}
        
        bg_color = background_color

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        screenspace_points = torch.zeros_like(gs.get_xyz, dtype=gs.get_xyz.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass
        

        raster_settings = GaussianRasterizationSettings_rade(
            image_height=int(viewpoint_camera.height),
            image_width=int(viewpoint_camera.width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=1.0,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform.float(),
            sh_degree=sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,

            #rade-GS的默认参数
            kernel_size = 0.0,
            require_coord = True,
            require_depth = True,

        )
        rasterizer = GaussianRasterizer_rade(raster_settings=raster_settings)

        means3D = gs.get_xyz

        means2D = screenspace_points.contiguous().float()
        opacity = gs.get_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        scales = gs.get_scaling 

        rotations = gs.get_rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None

        shs = gs.get_features


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
            "depth_pred": rendered_expected_depth,
            "radii": radii,
            "visibility_filter" : radii > 0,
            "viewspace_points": screenspace_points,

        }
        return ret
def test_rendering_speed(batch):
    view_num = batch['c2w_output'].shape[1]
    cam_list = []
    gs_model = batch["gs"][0]
    for i in range(view_num):
        c2w = batch['c2w_output'][0,i]#不能用test view

        FOV = batch['FOV'][0]
        bg = batch['background_color'][0]
        cam = Camera.from_c2w(c2w, FOV,  batch['resolution'][0])
        cam_list.append(cam)

    duration = []

    for i in range(3):
        for cam in tqdm(cam_list):
            start = time.time()
            render_pkg = forward_single_view(gs_model, cam, bg, sh_degree=3)
            duration.append(time.time()-start)
    fps = 1/np.mean(duration)
    return  fps
def infer(cfg):
    cfg.data.data.up_sample = cfg.system.up_sample

    dataset = igs.find(cfg.data.data_cls)(cfg.data.data,training=False)
    dataloader = DataLoader(dataset,
                        batch_size=cfg.opt.eval_batch_size, 
                        num_workers=0, # warn hardcode
                        shuffle=False,#False
                        collate_fn=dataset.collate
                    )
    
    if cfg.opt.free_view:
        dataset.get_spiral()
        free_views = []
    
    if cfg.opt.refine_gs:
        # build key frame
        dataset.build_refine_dataset(cfg.opt.eval_batch_size)



    model = IGS(cfg.system).to(device)
    if cfg.opt.stream_eval:
        model.set_stream_eval()
    elif cfg.opt.stream_eval_batch:
        model.stream_eval_batch = True
    ckpt = torch.load(cfg.opt.resume, map_location='cpu')
    


    state_dict = model.state_dict()
    for k, v in ckpt.items():
        if k in state_dict: 
            if state_dict[k].shape == v.shape:
                state_dict[k].copy_(v)
    print("load model ckpt done")

    print("batched inference step", len(dataloader))
    print("current workspace", cfg.opt.workspace)
    psnrs = []
    mask_num = []#number of masked
    points_num = [] #number of keyframe points
    out_images = []
    pbar = tqdm(enumerate(dataloader), total = len(dataloader))
    big_mask = None
    perframe_times = []
    AGM_times = []

    if model.render.cfg.sh_degree != cfg.data.data.max_sh_degree:
        # align training sh_degree to eval sh_degree
        print(f"reset sh_degree to {cfg.data.data.max_sh_degree}")
        model.render.cfg.sh_degree = cfg.data.data.max_sh_degree

    for idx, batch in pbar:
        batch = todevice(batch)

        # prepare input to AGM net
        if idx==0:
            start_gs = batch["gs"][0]
            print("start point num:",start_gs.get_xyz.shape[0])
            if cfg.opt.free_view:
                gs_model = igs.find(cfg.opt.gs_model_cls)(cfg.opt.gs_model)
                gs_model.load_fromstream(start_gs, cfg.opt.training_lr, refine_item=cfg.opt.refine_item, mask=big_mask)
                gs_model.save_ply(f'{cfg.opt.workspace}/gs/{0}.ply')
            start_depth = batch["depth"]
            start_history = None
            fps = test_rendering_speed(batch)
        else:            
            start_depth = depth_pred.repeat(batch["cur_images_input"].shape[0],1,1,1)
            if batch['keyframe'][0].cpu().numpy()==1:
                start_gs = stream_gs
            batch["depth"] = start_depth # from last
            batch["gs"] = [start_gs]

        start_time = time.time()

        #AGM net inference
        out = model(batch)

        duration = time.time()-start_time# the duration of AGM-Net inference

        perframe_times+=[duration/batch['images_output'].shape[0] for i in range(batch['images_output'].shape[0])]
        AGM_times+=[duration]

        pred_images = out['images_pred'][:,0:1]
        out_images.extend([img for img in pred_images.detach().squeeze(1).cpu()])

        # compute PSNR
        gt_images = batch['images_output'][:,0:1]
        pred_images = torch.clamp(pred_images,0,1)
        psnr = -10 * torch.log10(torch.mean((pred_images.detach() - gt_images) ** 2, dim=(1,2,3,4)))
        psnrs += psnr.tolist()

        depth_pred = out['depth_pred'][-1:,1:].squeeze(2)
        stream_gs = out["3dgs"][-1]
        mask_num.append(len(stream_gs.mask))
        points_num.append(stream_gs.get_xyz.shape[0])


        if cfg.opt.refine_gs:
            # key-frame guided streaming
            start_time = time.time()
            
            # 定義目前的 key
            current_key = (idx+1)*cfg.opt.eval_batch_size

            # [修改] 檢查 key 是否在 dataset 中 (不管是 set 還是 dict 語法都一樣)
            if current_key in dataset.refine_dataset:
                
                # === [新增] Online 讀取資料: 只讀取這一組 ===
                print(f"Loading refine data for frame {current_key}...")
                refine_data = dataset.get_refine_data(current_key)
                # ==========================================

                with torch.enable_grad():

                    viewpoint_cam = None

                    gs_model = igs.find(cfg.opt.gs_model_cls)(cfg.opt.gs_model)
                    if cfg.opt.use_anchor:
                        gs_model.anchor_idx = model.pre_compute_states["fps_idx_precompute"]
                    opt_all = True

                    #convert stream gs model to a GaussianModel 
                    gs_model.load_fromstream(stream_gs, cfg.opt.training_lr, refine_item=cfg.opt.refine_item, mask=big_mask)
                    if cfg.opt.use_densify:
                        gs_model.max_num = cfg.opt.max_num
                        model.reset_pre_compute_states()

                    # iterations = cfg.opt.refine_iterations
                    # if (idx+1) % 50 ==0:
                    #     iterations = 150

                    for iteration in range(cfg.opt.refine_iterations):
                        if not viewpoint_cam:
                            # [修改] 從剛剛讀取的 refine_data 複製資料
                            viewpoint_cam = refine_data["c2ws"].copy()
                            viewpoint_img = refine_data["images"].copy()

                        # [修改] 從 viewpoint_cam/img 取出資料 (這裡邏輯不變)
                        pick = randint(0, len(viewpoint_cam)-1)
                        c2w = viewpoint_cam.pop(pick).to("cuda")
                        gt_image = viewpoint_img.pop(pick).to("cuda")
                        
                        # [修改] 從 refine_data 讀取 FOV 和 bg
                        FOV = refine_data["FOV"].to("cuda")
                        bg = refine_data["bg"].to("cuda")

                        cam = Camera.from_c2w(c2w, FOV,  gt_image.shape[-2:])
                        if cfg.opt.use_ntc:
                            gs_model.query_ntc()
                        elif cfg.opt.use_anchor:
                            gs_model.query_ir_grid()
                        render_pkg = forward_single_view(gs_model, cam, bg, sh_degree=cfg.data.data.max_sh_degree)
                        render_image, radii, visibility_filter, viewspace_point_tensor = render_pkg["images_pred"], render_pkg["radii"], render_pkg["visibility_filter"], render_pkg["viewspace_points"]
                        psnr = -10 * torch.log10(torch.mean((render_image.detach() - gt_image) ** 2))
                        Ll1_render = l1_loss(render_image, gt_image)

                        rgb_loss = cfg.opt.lambda_l1 * Ll1_render +(1-cfg.opt.lambda_l1) * (1.0 - ssim(render_image, gt_image.unsqueeze(0), size_average=False))
                        loss = rgb_loss
                        loss.backward()
                        with torch.no_grad():
                            if cfg.opt.use_densify:
                                
                                if iteration < cfg.opt.densify_until_iter:
                                    # Keep track of max radii in image-space for pruning
                                    gs_model.max_radii2D[visibility_filter] = torch.max(gs_model.max_radii2D[visibility_filter], radii[visibility_filter])
                                    gs_model.add_densification_stats(viewspace_point_tensor, visibility_filter)

                                    if iteration > cfg.opt.densify_from_iter and iteration % cfg.opt.densification_interval == 0:
                                        size_threshold =  None
                                        control_max = True
                                        if "control_max" in cfg.opt:
                                            control_max = cfg.opt.control_max
                                        #max points bounded densify
                                        gs_model.densify_and_prune(cfg.opt.densify_grad_threshold, 0.005, batch["radius"][0], size_threshold,control_max=control_max)
                
                            gs_model.optimizer.step()
                            gs_model.optimizer.zero_grad(set_to_none = True)

                    stream_gs = gs_model.convert2stream()
                
                # === [新增] 釋放記憶體 ===
                del refine_data
                del viewpoint_cam
                del viewpoint_img
                # 清理迴圈變數
                if 'render_pkg' in locals(): del render_pkg
                if 'render_image' in locals(): del render_image
                if 'loss' in locals(): del loss
                if 'rgb_loss' in locals(): del rgb_loss
                
                import gc
                gc.collect()
                torch.cuda.empty_cache() # 建議加上這行以確保 GPU/CPU 記憶體歸還
                # =======================

                c2w = batch["c2w_output"][0,0]                
                FOV = batch["FOV"][0]

                cam = Camera.from_c2w(c2w, FOV, batch['resolution'][0])
                render_pkg = forward_single_view(gs_model, cam, batch['background_color'][0], sh_degree=cfg.data.data.max_sh_degree)
                render_image, depth_pred_ = render_pkg["images_pred"], render_pkg["depth_pred"]

                render_image = torch.clamp(render_image,0,1)

                gt_images = gt_images[-1].detach()
                psnr = -10 * torch.log10(torch.mean((render_image - gt_images) ** 2))
                psnrs[-1] = psnr.item()

                out_images[-1] = render_image.detach().cpu()
                out["3dgs"][-1] = stream_gs

        if cfg.opt.free_view:
            #generate free view
            start_view = idx*cfg.opt.eval_batch_size
            for i in range(len(out["3dgs"])):
                gs = out["3dgs"][i]
                c2w = dataset.free_poses[start_view+i].to("cuda")
                # c2w = batch["c2w_output"][0,0]                

                # print(c2w)
                FOV = batch["FOV"][0]

                cam = Camera.from_c2w(c2w, FOV, batch['resolution'][0])
                gs_model = igs.find(cfg.opt.gs_model_cls)(cfg.opt.gs_model)
                gs_model.load_fromstream(gs, cfg.opt.training_lr, refine_item=cfg.opt.refine_item, mask=big_mask)

                render_pkg = forward_single_view(gs_model, cam, batch['background_color'][0], sh_degree=cfg.data.data.max_sh_degree)
                render_image, depth_pred_ = render_pkg["images_pred"], render_pkg["depth_pred"]
                free_views.append(render_image)

                gs_model.save_ply(f'{cfg.opt.workspace}/gs/{idx*cfg.opt.eval_batch_size+i+1}.ply')


        perframe_times[-1]+=time.time()-start_time # add key-frame refine time to per-frame time
        historry_state = out.get("motion_feature", None)
        
        # === [新增] 迴圈末尾釋放 gs_model ===
        if 'gs_model' in locals(): del gs_model
        
        # [新增] 釋放 out 字典，因為它包含了很多中間變數
        if 'out' in locals(): del out
        
        torch.cuda.empty_cache()
        # ==================================

        # [新增] 監控點雲數量
        if 'stream_gs' in locals():
             print(f"Frame {idx}: stream_gs points = {stream_gs.get_xyz.shape[0]}")




    print("avg psnrs:", np.mean(psnrs))
    total_time = pbar.format_dict['elapsed'] # use all the time, same with the paper report
    print("total time", total_time)
    print("total_AGM_times", np.sum(np.array(AGM_times)))
    print("per frame train time:", total_time/len(psnrs))
    result = {'psnr':{f"frame_{i+1}": psnrs[i] for i in range(len(psnrs))}, "avg":np.mean(psnrs), "total_time":total_time,"mask_num":mask_num, "points_num":points_num, "fps":fps,"per_frame_times":perframe_times,"AGM_times":AGM_times}
    resultes_path = f'{cfg.opt.workspace}/results.json'
    with open(resultes_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False,indent=4)
    os.makedirs(f'{cfg.opt.workspace}/eval_pred/', exist_ok=True)
    for idx, img in enumerate(out_images):
        torchvision.utils.save_image(img, f'{cfg.opt.workspace}/eval_pred/'+'{0:05d}'.format(idx+1)+'.png')
    
    if cfg.opt.free_view:
        os.makedirs(f'{cfg.opt.workspace}/free_views/', exist_ok=True)
        for idx, img in enumerate(free_views):
            torchvision.utils.save_image(img, f'{cfg.opt.workspace}/free_views/train_pred_images_{idx+1}.png')


def transpose_2_write(img_tensor):
    #b v c h w
    assert len(img_tensor.shape) == 5
    c = img_tensor.shape[2]
    img_np = img_tensor.detach().cpu().numpy() # [B, V, 3, output_size, output_size]
    img_np = img_np.transpose(0, 3, 1, 4, 2).reshape(-1, img_np.shape[1] * img_np.shape[3], 3) # [B*output_size, V*output_size, 3]
    return img_np

if __name__ == "__main__":
    import argparse
    import subprocess
    from igs.utils.config import ExperimentConfig, load_config
    # from tgs.data import CustomImageOrbitDataset
    from igs.utils.misc import todevice, get_device

    parser = argparse.ArgumentParser("Triplane Gaussian Splatting")
    parser.add_argument("--config", required=True, help="path to config file")
    parser.add_argument("--out", default="outputs", help="path to output folder")
    parser.add_argument("--cam_dist", default=1.9, type=float, help="distance between camera center and scene center")
    parser.add_argument("--image_preprocess", action="store_true", help="whether to segment the input image by rembg and SAM")
    args, extras = parser.parse_known_args()

    device = get_device()

    cfg: ExperimentConfig = load_config(args.config, cli_args=extras)
    
    resume_cfg: ExperimentConfig = load_config(cfg.opt.resume_cfg)
    cfg.system = resume_cfg.system
    
    os.makedirs(cfg.opt.workspace,exist_ok=True)
    # convert ExperimentConfig to DictConfig

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    # save as yaml
    with open(os.path.join(cfg.opt.workspace,'experiment_config.yaml'), 'w') as f:
        OmegaConf.save(config=cfg_dict, f=f)

    ic.disable()

    with torch.no_grad():
        infer(cfg)






