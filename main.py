# import tyro
import time
from dataclasses import dataclass, field
import os

os.environ['NCCL_BLOCKING_WAIT'] = '0'  # not to enforce timeout
os.environ["NCCL_P2P_DISABLE"] = '1'

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
from igs.utils.loss_utils import l1_loss, ssim, quaternion_loss
from igs.utils.general_utils import visualize_flow, build_rotation
import torch.nn as nn
import igs
from icecream import ic
from tqdm import tqdm
from kiui.lpips import LPIPS
from torchvision.utils import flow_to_image
from torch_cluster import fps, knn

def saveRuntimeCode(dst: str) -> None:
    if os.path.exists(dst):
        print("backup path exists, failed")
        return
    additionalIgnorePatterns = ['.git', '.gitignore']
    ignorePatterns = set()
    ROOT = '.'
    with open(os.path.join(ROOT, '.gitignore')) as gitIgnoreFile:
        for line in gitIgnoreFile:
            if not line.startswith('#'):
                if line.endswith('\n'):
                    line = line[:-1]
                if line.endswith('/'):
                    line = line[:-1]
                ignorePatterns.add(line)
    ignorePatterns = list(ignorePatterns)
    for additionalPattern in additionalIgnorePatterns:
        ignorePatterns.append(additionalPattern)

    log_dir = pathlib.Path(__file__).parent.resolve()

    shutil.copytree(log_dir, dst, ignore=shutil.ignore_patterns(*ignorePatterns),dirs_exist_ok=True)
    
    print('Backup Finished!')


@dataclass
class OptConfig:
    compute_environment: str = "LOCAL_MACHINE"
    debug: bool = False
    distributed_type: str = "MULTI_GPU"
    downcast_bf16: str = "no"
    machine_rank: int = 0
    main_training_function: str = "main"
    mixed_precision: str = "fp16"
    num_machines: int = 1
    num_processes: int = 4
    rdzv_backend: str = "static"
    same_network: bool = True
    tpu_env: list = field(default_factory=list)
    tpu_use_cluster: bool = False
    tpu_use_sudo: bool = False
    use_cpu: bool = False
    gradient_accumulation_steps: int = 1
    resume: Optional[str] = None
    resume_opt: Optional[str] = None

    data_mode: str = "s3"
    batch_size: int = 8
    lr: float = 4e-4
    num_workers: int = 4  # Assuming batch_size is 8
    num_epochs: int = 30
    workspace: str = "/workspace"

    lambda_lpips: float = 0
    lambda_ssim : float = 0
    lambda_flow : float = 0
    lambda_render_flow : float = 0
    lambda_rgb : float = 1
    lambda_gstream: float = 0
    lambda_rigid: float = 0.1

    gradient_clip: float = 1.0

    image_list: Any = ""
    background_color: Tuple[float, float, float] = field(
        default_factory=lambda: (1.0, 1.0, 1.0)
    )
    relative_pose: bool = False

    debug:bool = False
    project:str = "obja-lvis"
    exp_name:str = "test"

    start_epoch:int = 0

    flow_cls: str = ""
    flow: dict = field(default_factory=dict)

    different_lr: bool = False


def main(cfg, accelerator):    
    opt: OptConfig = parse_structured(OptConfig, cfg.opt)


    writer = None
    if accelerator.is_main_process and not opt.debug:
        writer = SummaryWriter(os.path.join(opt.workspace,"runs"))

    # model
    model = IGS(cfg.system)

    start_epoch = 0

    if opt.resume is not None:
        start_epoch = opt.start_epoch

        if opt.resume.endswith('safetensors'):
            ckpt = load_file(opt.resume, device='cpu')
        else:
            ckpt = torch.load(opt.resume, map_location='cpu')

        state_dict = model.state_dict()
        # print(state_dict.keys(), ckpt.keys())
        for k, v in ckpt.items():
            if k in state_dict: 
                if state_dict[k].shape == v.shape:
                    state_dict[k].copy_(v)
                else:
                    accelerator.print(f'[WARN] mismatching shape for param {k}: ckpt {v.shape} != model {state_dict[k].shape}, ignored.')
            else:
                accelerator.print(f'[WARN] unexpected param {k}: {v.shape}')
        accelerator.print("load ckpt ok")

    train_dataset = igs.find(cfg.data.data_cls)(cfg.data.data, training=True)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True, # false only debug
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=train_dataset.collate
    )

    test_dataset = igs.find(cfg.data.data_cls)(cfg.data.data,training=False)

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=test_dataset.collate
    )
 

    if opt.different_lr:
        all_params = {name: param for name, param in model.named_parameters() if param.requires_grad}

        temporal_net_params = {name: param for name, param in all_params.items() if "temporal_net" in name}

        params = [
            {'params': temporal_net_params.values(), 'lr': opt.lr},
            {'params': [param for name, param in all_params.items() if name not in temporal_net_params], 'lr': 0.1*opt.lr}
        ]
        optimizer = torch.optim.AdamW(params, weight_decay=0.05, betas=(0.9, 0.95))

    else:
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr, weight_decay=0.05, betas=(0.9, 0.95))

    total_steps = opt.num_epochs * len(train_dataloader)
    accelerator.print(opt.num_epochs,len(train_dataloader),total_steps)
    pct_start = 3000 / total_steps

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=opt.lr, total_steps=total_steps, pct_start=pct_start)

    if opt.resume is not None:

        if opt.resume_opt is not None:
            ckpt_opt = torch.load(opt.resume_opt, map_location='cpu')
            optimizer.load_state_dict(ckpt_opt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt_opt['scheduler_state_dict'])

    # accelerate
    model, optimizer, train_dataloader, test_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, test_dataloader, scheduler
    )
    
    # if opt.lambda_flow >0:
    #     flow_model = igs.find(opt.flow_cls)(opt.flow)

    #     flow_model = accelerator.prepare(flow_model)
    #     flow_model.eval()
    #     flow_model.requires_grad_(False)
    
    if opt.lambda_lpips >0:
        lpips_loss = LPIPS(net="vgg")
        lpips_loss.requires_grad_(False)
        lpips_loss = accelerator.prepare(lpips_loss)
        lpips_loss.requires_grad_(False)

   
    for epoch in range(start_epoch, opt.num_epochs):
        # train
        model.train()
        total_loss = 0
        total_psnr = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch}", disable=not accelerator.is_local_main_process)
        
        for i, data in enumerate(train_dataloader):

            with accelerator.accumulate(model):
                optimizer.zero_grad()
                loss = 0
                step_ratio = (epoch + i / len(train_dataloader)) / opt.num_epochs

                image_shape = data["cur_images_input"].shape
                cur_images_input = data["cur_images_input"].reshape((-1, *image_shape[-3:]))
                next_images_input = data["next_images_input"].reshape((-1, *image_shape[-3:]))
                
                cur_images_input = F.interpolate(cur_images_input, size=(cfg.data.data.input_height, cfg.data.data.input_width), mode='bilinear', align_corners=False)
                next_images_input = F.interpolate(next_images_input, size=(cfg.data.data.input_height, cfg.data.data.input_width), mode='bilinear', align_corners=False)
                data["cur_images_input"] = cur_images_input.reshape((*image_shape[:-2], cfg.data.data.input_height, cfg.data.data.input_width))

                data["next_images_input"] = next_images_input.reshape((*image_shape[:-2], cfg.data.data.input_height, cfg.data.data.input_width))
                data['images_output'].to("cpu")
                out = model(data)
                pred_images = out['images_pred'] # [B, V, C, output_size, output_size]

                gt_images = data['images_output'].to("cuda") # [B, V, 3, output_size, output_size], ground-truth novel views

               
                if opt.lambda_rgb >0:

                    loss_mse = l1_loss(pred_images, gt_images)
                    out["loss_mse"] = loss_mse.item()
                    loss = loss + opt.lambda_rgb * loss_mse

                if opt.lambda_ssim >0:
                    img1 = rearrange(pred_images, " B V C H W -> (B V) C H W")
                    img2 = rearrange(gt_images, " B V C H W -> (B V) C H W")
                    ssim_value, ssim_map = ssim(img1, img2)

                    loss_ssim = 1.0 - ssim_value
                    out['loss_ssim'] = loss_ssim.item()
                    loss = loss + opt.lambda_ssim*loss_ssim

                if opt.lambda_lpips > 0:
                    loss_lpips = lpips_loss(

                        # downsampled to at most 256 to reduce memory cost
                        F.interpolate(gt_images.view(-1, 3,cfg.data.data.output_height, cfg.data.data.output_width) * 2 - 1, (256, 256), mode='bilinear', align_corners=False), 
                        F.interpolate(pred_images.view(-1, 3,cfg.data.data.output_height, cfg.data.data.output_width) * 2 - 1, (256, 256), mode='bilinear', align_corners=False),
                    ).mean()
                    out['loss_lpips'] = loss_lpips.item()
                    loss = loss + opt.lambda_lpips * loss_lpips
                

                try:
                    accelerator.backward(loss)
                except:
                    print("idx", data["idx"])
                    print("loss", loss)
                    for k in out:
                        if "loss" in k:
                            print(k, out[k])
                    accelerator.save_state(f'{opt.workspace}/error')
                    raise RuntimeError("Loss backward")

                with torch.no_grad():
                    psnr = -10 * torch.log10(torch.mean((pred_images.detach() - gt_images) ** 2))


                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), opt.gradient_clip)
                

                optimizer.step()
                scheduler.step()
                total_loss += loss.detach()
                total_psnr += psnr.detach()

            if accelerator.is_main_process:

                with torch.no_grad():
                    if i>0 and i%10 ==0:
                        progress_bar.set_postfix({'loss': loss.item()})
                        progress_bar.update(10)

                    if i % 100 == 0:

                        
                        mem_free, mem_total = torch.cuda.mem_get_info()  

                        loss_dict = {k:v for k,v in out.items() if "loss" in k}
                        loss_output = ""
                        for k in loss_dict:
                            loss_output += f"{k}:{loss_dict[k]:6f} "

                        print(f"[INFO] {i}/{len(train_dataloader)} mem: {(mem_total-mem_free)/1024**3:.2f}/{mem_total/1024**3:.2f}G lr: {scheduler.get_last_lr()[0]:.7f} step_ratio: {step_ratio:.4f} loss: {loss.item():.6f}", loss_output)

                        if writer is not None:
                            writer.add_scalar("train_loss", loss.item(), epoch*len(train_dataloader)+i) #日志中记录x在第step i 的值
            
                    # save log images
                    # if i % 500 == 0:

                    #     if len(gt_images.shape)==6:
                    #         gt_images = gt_images[:,-1]
                    #         pred_images = pred_images[:,-1]

                    #     gt_images = gt_images.detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                    #     gt_images = gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images.shape[1] * gt_images.shape[4], 3) # [B*output_size, V*output_size, 3]
                    #     kiui.write_image(f'{opt.workspace}/train_gt/train_gt_images_{epoch}_{i}.jpg', gt_images)
   

                    #     pred_images = pred_images.detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                    #     pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[4], 3)
                    #     pred_images = np.clip(pred_images, 0.0 ,1.0)
                    #     kiui.write_image(f'{opt.workspace}/train_pred/train_pred_images_{epoch}_{i}.jpg', pred_images)
                        
                    #     error_map = abs(pred_images - gt_images).mean(axis=-1)[...,np.newaxis]

                    #     kiui.write_image(f'{opt.workspace}/error_map/err_map_{epoch}_{i}.jpg', error_map)
                        
                    #     if opt.lambda_ssim >0:
                    #         ssim_map = torch.linalg.norm(ssim_map +1 /2, dim=1, keepdim=True)
                    #         ssim_map = rearrange(ssim_map, "(B V) C H W -> B V C H W", V =cfg.data.data.num_output_views).detach().cpu().numpy()
                            
                    #         ssim_map = ssim_map.transpose(0, 3, 1, 4, 2).reshape(-1, ssim_map.shape[1] * ssim_map.shape[4], 1)
                    #         ssim_map = np.clip(ssim_map, 0.0, 1.0)
                    #         kiui.write_image(f'{opt.workspace}/ssim_map/ssim_map_{epoch}_{i}.jpg', ssim_map)
                               
        total_loss = accelerator.gather_for_metrics(total_loss).mean()
        total_psnr = accelerator.gather_for_metrics(total_psnr).mean()#聚合所有process上的指标
        if accelerator.is_main_process:
            total_loss /= len(train_dataloader)
            total_psnr /= len(train_dataloader)
            accelerator.print(f"[train] epoch: {epoch} loss: {total_loss.item():.6f} psnr: {total_psnr.item():.4f}")
            if writer is not None:
                writer.add_scalar("total_loss", total_loss.item(), epoch) #日志中记录x在第step i 的值
                writer.add_scalar("total_psnr", total_psnr.item(), epoch) #日志中记录x在第step i 的值
                
                for k,v in out.items():
                     if "loss" in k:
                        writer.add_scalar("k", v, epoch)
        # checkpoint
        # if epoch % 10 == 0 or epoch == opt.num_epochs - 1:
        accelerator.wait_for_everyone()
        ckpt_path = os.path.join(opt.workspace,str(epoch))
        accelerator.save_model(model, ckpt_path,safe_serialization=False)

        checkpoint = {
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        }
        torch.save(checkpoint, os.path.join(ckpt_path,"optim.pth"))

        # eval
        with torch.no_grad():
            model.eval()
            total_psnr = 0
            for i, data in enumerate(test_dataloader):

                image_shape = data["cur_images_input"].shape
                cur_images_input = data["cur_images_input"].reshape((-1, *image_shape[-3:]))
                next_images_input = data["next_images_input"].reshape((-1, *image_shape[-3:]))
                
                cur_images_input = F.interpolate(cur_images_input, size=(cfg.data.data.input_height, cfg.data.data.input_width), mode='bilinear', align_corners=False)
                next_images_input = F.interpolate(next_images_input, size=(cfg.data.data.input_height, cfg.data.data.input_width), mode='bilinear', align_corners=False)
                data["cur_images_input"] = cur_images_input.reshape((*image_shape[:-2], cfg.data.data.input_height, cfg.data.data.input_width))

                data["next_images_input"] = next_images_input.reshape((*image_shape[:-2], cfg.data.data.input_height, cfg.data.data.input_width))
 

                out = model(data)
                pred_images = out['images_pred'] # [B, V, C, output_size, output_size]


                gt_images = data['images_output'] # [B, V, 3, output_size, output_size], ground-truth novel views

                bg_color = data["background_color"]
                psnr = -10 * torch.log10(torch.mean((pred_images.detach() - gt_images) ** 2))

                # psnr = out['psnr']
                total_psnr += psnr.detach()
                
                # save some images
                if accelerator.is_main_process:
                    if len(gt_images.shape)==6:
                        gt_images=gt_images[:,0]
                        pred_images = pred_images[:,0]
                    
                    #only save first of a batch
                    gt_images = gt_images[:1]
                    pred_images = pred_images[:1]

                    gt_images = gt_images.detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                    gt_images = gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images.shape[1] * gt_images.shape[4], 3) # [B*output_size, V*output_size, 3]
                    kiui.write_image(f'{opt.workspace}/eval_gt/eval_gt_images_{epoch}_{i}.jpg', gt_images)

                    pred_images = pred_images.detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                    pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[4], 3)
                    kiui.write_image(f'{opt.workspace}/eval_pred/eval_pred_images_{epoch}_{i}.jpg', pred_images)



            total_psnr = accelerator.gather_for_metrics(total_psnr).mean()
            if accelerator.is_main_process:
                total_psnr /= len(test_dataloader)
                accelerator.print(f"[eval] epoch: {epoch} psnr: {total_psnr.item():.4f}")
                if writer is not None:
                    writer.add_scalar("eval_psnr", total_psnr.item(), epoch) #日志中记录x在第step i 的值



if __name__ == "__main__":
    import argparse
    import subprocess
    from igs.utils.config import ExperimentConfig, load_config
    from igs.utils.misc import todevice, get_device

    parser = argparse.ArgumentParser("Triplane Gaussian Splatting")
    parser.add_argument("--config", required=True, help="path to config file")
    parser.add_argument("--out", default="outputs", help="path to output folder")
    parser.add_argument("--cam_dist", default=1.9, type=float, help="distance between camera center and scene center")
    parser.add_argument("--image_preprocess", action="store_true", help="whether to segment the input image by rembg and SAM")
    args, extras = parser.parse_known_args()

    device = get_device()

    cfg: ExperimentConfig = load_config(args.config, cli_args=extras)

    accelerator = Accelerator(
        mixed_precision=cfg.opt.mixed_precision,
        gradient_accumulation_steps=cfg.opt.gradient_accumulation_steps,
    )


    if accelerator.is_main_process:

        os.makedirs(cfg.opt.workspace,exist_ok=True)


        # 将 ExperimentConfig 对象转换为 DictConfig
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        # 保存为 YAML 文件
        with open(os.path.join(cfg.opt.workspace,'experiment_config.yaml'), 'w') as f:
            OmegaConf.save(config=cfg_dict, f=f)

        saveRuntimeCode(os.path.join(cfg.opt.workspace,"backup"))

    np.random.seed(6)
    torch.manual_seed(1111)
    torch.cuda.manual_seed(2222)
    torch.cuda.manual_seed_all(3333)
    ic.disable()
    main(cfg, accelerator)






    
