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
import kiui
from igs.models.gs import GaussianModel, load_ply
from icecream import ic
import gc

@dataclass
class N3dDatasetConfig:
    background_color: Tuple[float, float, float] = field(
        default_factory=lambda: (1.0, 1.0, 1.0)
    )


    data_path:str = ""
    bbox_path: str = "bbox.json"
    root_dir:str =""
    num_input_views:int = 16
    num_output_views:int = 20



    output_height:int = 1014
    output_width:int = 1352

    input_height:int = 1024
    input_width:int = 1024


    gs_mode:str = "3dgs_rade"
    iter:str = "10000_compress"

    need_rays: bool = True
    need_flow: bool = True
    up_sample: bool = False
    use_group: bool = False
    use_gstream: bool =False
    max_sh_degree: int = 1

class N3dDataset(Dataset):
    def __init__(self, cfg:Any, training=True):
        super().__init__()

        self.cfg: N3dDatasetConfig = parse_structured(N3dDatasetConfig, cfg)

        # self.opt = opt
        self.training = training

        n3d_path = os.path.join(cfg.root_dir, self.cfg.data_path) 

        with open(n3d_path, 'r') as f:
            n3d_paths = json.load(f)
            if self.training:
                self.items = n3d_paths['train']
            else:
                self.items = n3d_paths['val']

        bbox_path = os.path.join(cfg.root_dir, self.cfg.bbox_path)
        with open(bbox_path, 'r') as f:
            bbox_path = json.load(f)
            self.bboxs = bbox_path

        self.background_color = torch.as_tensor(self.cfg.background_color)

    def __len__(self):
        return len(self.items)
        


    def __getitem__(self, idx):
        '''
        {
            "input_cur_images":[V_input,H,W]
            "output_next_images":[V_output,H,W]
            "input_c2ws":[V_input,H,W]
            "output_c2ws":[V_output,H,W]
            "FOV"[2] # For the same scene, the intrinsic camera parameters are identical across all viewpoints.
        }
        '''
        scene_name = self.items[idx]["scene_name"]
        cur_frame = self.items[idx]["cur_frame"]
        next_frame = self.items[idx]["next_frame"]
        cur_frame_dir = os.path.join(self.cfg.root_dir, scene_name, cur_frame)
        next_frame_dir = os.path.join(self.cfg.root_dir, scene_name, next_frame)

        cameras_json_path = os.path.join(cur_frame_dir,self.cfg.gs_mode,"cameras.json")
        with open(cameras_json_path) as f:
            cameras_data = json.load(f)
        

        cam_centers = []
        for cam in cameras_data:
            cam_centers.append(np.array(cam["position"])[...,np.newaxis])
        scene_info = getNerfppNorm(cam_centers)

        translate = scene_info["translate"]
        radius = scene_info["radius"]

        bbox = torch.tensor(self.bboxs[scene_name]).to(torch.float)
        if self.training:

            if self.cfg.use_group:
                group_json_path = os.path.join(self.cfg.root_dir, scene_name, "group.json")
                with open(group_json_path) as f:
                    group_data = json.load(f)

                # Randomly select one from each group as input.
                selected_numbers = [random.choice(lst) for lst in group_data]
                all_numbers = [num for lst in group_data for num in lst]
                remaining_numbers = [num for num in all_numbers if num not in selected_numbers]
                additional_numbers = random.sample(remaining_numbers, self.cfg.num_output_views - 4)

                vids = selected_numbers+additional_numbers
                
            else:
                vids = np.arange(self.cfg.num_output_views).tolist() # fixed here

        else:
            vids = [3, 7, 1, 4, 8, 0]

        cur_images = []
        next_images = []
        depth_images = []
        c2ws = []

        flows = []
        results = {}

        for vid in vids:
            image_name = cameras_data[vid]["img_name"]
            image_name_id = str(vid).zfill(5) #renderings are named by id

            cur_image_path = os.path.join(cur_frame_dir, self.cfg.gs_mode,"train",f"ours_{self.cfg.iter}","gt", image_name_id+".png")
            next_image_path = os.path.join(next_frame_dir, self.cfg.gs_mode,"train",f"ours_{self.cfg.iter}","gt", image_name_id+".png")
            depth_image_path = os.path.join(cur_frame_dir, self.cfg.gs_mode,"train",f"ours_{self.cfg.iter}","depth_expected_mm", image_name_id+".png")

            cur_image = torch.from_numpy(np.array(Image.open(cur_image_path))/255.0).permute(2,0,1).to(torch.float)
            next_image = torch.from_numpy(np.array(Image.open(next_image_path))/255.0).permute(2,0,1).to(torch.float)
            depth_image = torch.from_numpy(np.array(Image.open(depth_image_path))/1000.0).to(torch.float)

            c2w = np.zeros((4, 4))
            c2w[:3,:3] = np.array(cameras_data[vid]["rotation"])
            c2w[:3,3] = np.array(cameras_data[vid]["position"])
            c2w[3,3] = 1
            c2w = torch.from_numpy(c2w).to(torch.float)

            fx = cameras_data[vid]["fx"]
            fy =cameras_data[vid]["fy"]
            width = cameras_data[vid]["width"]
            height = cameras_data[vid]["height"]

            FovX = focal2fov(fx, width)
            FovY = focal2fov(fy, height)

            cur_images.append(cur_image)
            next_images.append(next_image)
            depth_images.append(depth_image)
            c2ws.append(c2w)

                
        cur_images = torch.stack(cur_images, dim=0) # [V, C, H, W]
        next_images = torch.stack(next_images, dim=0) # [V,C, H, W]
        depth_images = torch.stack(depth_images, dim=0)
        c2ws = torch.stack(c2ws, dim=0) # [V, 4, 4]


        cur_images_input = cur_images[:self.cfg.num_input_views].clone()
        next_images_input = next_images[:self.cfg.num_input_views].clone()

        depth_images_input = depth_images[:self.cfg.num_input_views].clone()
        
        c2ws_input = c2ws[:self.cfg.num_input_views].clone()
        
        #define path of gaussian
        gs_path = os.path.join(cur_frame_dir, self.cfg.gs_mode,"point_cloud",f"iteration_{self.cfg.iter}","point_cloud.ply")
        results["gs_path"] = gs_path


        results['cur_images_input'] = cur_images_input # [2,V, C, output_size, output_size]     
        results['next_images_input'] = next_images_input # [2,V, C, output_size, output_size]      

        results['images_output'] = next_images # [V, C, output_size, output_size]

        results["depth"] = depth_images_input

        #camera info 
        results['c2w_output'] = c2ws
        results['c2w_input'] = c2ws_input
        results['FOV'] = torch.tensor([FovX, FovY], dtype=torch.float32)
        results["background_color"] = self.background_color

        output_height, output_width = next_images_input.shape[-2:]
        results["resolution"] = torch.tensor([output_height, output_width])
        results["idx"] = idx

        results["radius"] = radius
        results["translate"] = translate
        results["bounding_box"] = bbox

        if self.cfg.need_rays:
            H = int(self.cfg.input_height / 8)
            W = int(self.cfg.input_width / 8)
            if self.cfg.up_sample:
                H, W = H*2, W*2
            fx , fy = fov2focal(FovX, W), fov2focal(FovY, H) 
            i, j = torch.meshgrid(
                torch.arange(W, dtype=torch.float32) + 0.5,
                torch.arange(H, dtype=torch.float32) + 0.5,
                indexing="xy",
            )

            directions: Float[Tensor, "H W 3"] = torch.stack(
                [(i - W/2) / fx, (j - H/2) / fy, torch.ones_like(i)], -1
            )
            directions = F.normalize(directions, p=2.0, dim=-1)
            results["local_rays"] = directions #local dir here

            dirs = c2ws_input[:,:3,:3]@ directions.view(-1,3).permute(1,0).unsqueeze(0)

            ori = c2ws_input[:,:3,3].unsqueeze(-1).repeat_interleave(int(H*W), dim=-1)

            rays = torch.cat([ori, dirs], dim=1)
            rays = rearrange(rays, " B D (H W) -> B H W D",H=H)
            results["rays"] = rays

        del rays, bbox, translate, radius, c2ws, c2ws_input, cur_images, next_images, depth_images, cam_centers, scene_info # avoid memory leak
        gc.collect()

        return results

    def collate(self, batch):
        batch = torch.utils.data.default_collate(batch)
        gs_list = []
        bounding_box_list = []
        points_list = []

        for gs_path in batch["gs_path"]:
            gs = load_ply(gs_path, max_sh_degree=self.cfg.max_sh_degree)
            points = gs.get_xyz


            gs_list.append(gs)
            points_list.append(points)

        batch.update({"gs": gs_list, "points": points_list})


        del gs_list, points_list
        gc.collect()
        return batch

