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
# from core.options import Options
# from core.utils import get_rays, grid_distortion, orbit_camera_jitter


@dataclass
class N3dDatasetConfig:
    background_color: Tuple[float, float, float] = field(
        default_factory=lambda: (1.0, 1.0, 1.0)
    )


    data_path:str = ""
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
    # eval_type: Optional[str] = None

    up_sample: bool = False

    use_group: bool = False

    use_gstream: bool =False
class N3dDatasetMultiframe(Dataset):

    # def _warn(self):
    #     raise NotImplementedError('this dataset is just an example and cannot be used directly, you should modify it to your own setting! (search keyword TODO)')
    '''
    通过深度图得到真实的pcd
    '''
    def __init__(self, cfg:Any, training=True):
        super().__init__()

        self.cfg: N3dDatasetConfig = parse_structured(N3dDatasetConfig, cfg)

        # self.opt = opt
        self.training = training

        # TODO: remove this barrier
        # self._warn()

        # TODO: load the list of objects for training
        # self.items = []
        # with open('TODO: file containing the list', 'r') as f:
        #     for line in f.readlines():
        #         self.items.append(line.strip())

        # # naive split
        # if self.training:
        #     self.items = self.items[:-self.opt.batch_size]
        # else:
        #     self.items = self.items[-self.opt.batch_size:]


        n3d_path = os.path.join(cfg.root_dir, self.cfg.data_path) # you can define your own split

        with open(n3d_path, 'r') as f:
            n3d_paths = json.load(f)
            if self.training:
                self.items = n3d_paths['train']
            else:
                self.items = n3d_paths['val']

       
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
            "FOV"[2] 不考虑不同相机内参不同的情况

            同一个场景，所有帧的相机都是一样的
        }
        '''
        scene_name = self.items[idx]["scene_name"]
        cur_frame_list = self.items[idx]["cur_frame"]
        next_frame_list = self.items[idx]["next_frame"]
        cur_frame_dir_0 = os.path.join(self.cfg.root_dir, scene_name, cur_frame_list[0])
        # next_frame_dir = os.path.join(self.cfg.root_dir, scene_name, next_frame)

        # print(idx, cur_frame_list)

        cameras_json_path = os.path.join(cur_frame_dir_0,self.cfg.gs_mode,"cameras.json")
        with open(cameras_json_path) as f:
            cameras_data = json.load(f)
        
        #读取radius和center
        cam_centers = []
        for cam in cameras_data:
            cam_centers.append(np.array(cam["position"])[...,np.newaxis])

        scene_info = getNerfppNorm(cam_centers)

        translate = scene_info["translate"]
        radius = scene_info["radius"]

        if self.training:

            if self.cfg.use_group:
                group_json_path = os.path.join(self.cfg.root_dir, scene_name, "group.json")
                with open(group_json_path) as f:
                    group_data = json.load(f)

                # 从每个列表中随机选取一个数字
                selected_numbers = [random.choice(lst) for lst in group_data]
                
                # 将所有列表中的数字合并到一个列表中
                all_numbers = [num for lst in group_data for num in lst]
                
                # 移除已经选中的数字
                remaining_numbers = [num for num in all_numbers if num not in selected_numbers]
                
                # 从剩下的数字中随机选取k个数字
                additional_numbers = random.sample(remaining_numbers, self.cfg.num_output_views - 4)

                vids = selected_numbers+additional_numbers
                
                # vids = [0]
                # vids = np.random.permutation(len(cameras_data))[:self.cfg.num_output_views].tolist()
            else:
                vids = np.arange(self.cfg.num_output_views).tolist() #先固定住，试一下前10个

                # ic(vids)
                # vids=[0,1,2,3]
                # vids =[ random.choice(list_dat) for list_dat in group_data]
                # vids = vids+

        else:
            vids = [13, 1, 8, 4, 2, 0]
            # vids = np.arange(self.cfg.num_output_views).tolist()

        multi_cur_images = []
        multi_next_images = []
        c2ws = []
        # multi_flows = []

        multi_cur_images_input = []
        multi_next_images_input = []
        
        depth_images_input = []#这个depth只能拿第一帧的
        
        c2ws_input = []
        results = {}

        for tid in range(len(cur_frame_list)):
            cur_frame_dir = os.path.join(self.cfg.root_dir, scene_name, cur_frame_list[tid])
            next_frame_dir = os.path.join(self.cfg.root_dir, scene_name, next_frame_list[tid])

            cur_images = []
            next_images = []
            depth_images = []
            c2ws_list = []
            # flows = []

            for vid in vids:
                image_name = cameras_data[vid]["img_name"]
                image_name_id = str(vid).zfill(5) #render后用的是id命名的
                cur_image_path = os.path.join(cur_frame_dir, self.cfg.gs_mode,"train",f"ours_{self.cfg.iter}","renders", image_name_id+".png")
                # next_image_path = os.path.join(next_frame_dir, "images", image_name+".png")
                next_image_path = os.path.join(next_frame_dir, self.cfg.gs_mode,"train",f"ours_{self.cfg.iter}","gt", image_name_id+".png")
                depth_image_path = os.path.join(cur_frame_dir, self.cfg.gs_mode,"train",f"ours_{self.cfg.iter}","depth_expected_mm", image_name_id+".png")

                #暂时不进行size变换
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
                c2ws_list.append(c2w)

                # if self.training:
                # flow_path = os.path.join(cur_frame_dir, self.cfg.gs_mode,"train",f"ours_{self.cfg.iter}","flows", f"cam_{vid}", f"{next_frame.split('_')[-1]}.npy")
                # flow = torch.from_numpy(np.load(flow_path)).to(torch.float)
                # flows.append(flow)
                
            cur_images = torch.stack(cur_images, dim=0) # [V, C, H, W]
            next_images = torch.stack(next_images, dim=0) # [V,C, H, W]
            # flows = torch.stack(flows, dim=0) 
            # c2w = np.linalg.inv(w2c)

            cur_images_input = cur_images[:self.cfg.num_input_views].clone()
            next_images_input = next_images[:self.cfg.num_input_views].clone()
            
            if tid ==0:
                depth_images = torch.stack(depth_images, dim=0)
                depth_images_input = depth_images[:self.cfg.num_input_views].clone()

                c2ws = torch.stack(c2ws_list, dim=0) # [V, 4, 4]
                # print("c2ws", c2ws.shape)
                c2ws_input = c2ws[:self.cfg.num_input_views].clone()

            multi_cur_images.append(cur_images)
            multi_cur_images.append(cur_images)
            multi_next_images.append(next_images)
            # multi_c2ws.append(c2ws)
            # multi_flows.append(flows)

            multi_cur_images_input.append(cur_images_input)
            multi_next_images_input.append(next_images_input)
            # multi_depth_images_input.append(depth_images_input)
            # multi_c2ws_input.append(c2ws_input)
        # 将所有数据堆叠起来
        multi_cur_images = torch.stack(multi_cur_images, dim=0)  # [T, V, C, H, W]
        multi_next_images = torch.stack(multi_next_images, dim=0)  # [T, V, C, H, W]
        # multi_c2ws = torch.stack(multi_c2ws, dim=0)  # [T, V, 4, 4]
        # multi_flows = torch.stack(multi_flows, dim=0)  # [T, V, 2, H, W]

        multi_cur_images_input = torch.stack(multi_cur_images_input, dim=0)  # [T, V, C, H, W]
        multi_next_images_input = torch.stack(multi_next_images_input, dim=0)  # [T, V, C, H, W]
        # multi_depth_images_input = torch.stack(multi_depth_images_input, dim=0)  # [T, V, H, W]
        # multi_c2ws_input = torch.stack(multi_c2ws_input, dim=0)


        
        #load Gaussian
        gs_path = os.path.join(cur_frame_dir_0, self.cfg.gs_mode,"point_cloud",f"iteration_{self.cfg.iter}","point_cloud.ply")
        results["gs_path"] = gs_path
        # results["gs"] = gs

        # stream_path = os.path.join(cur_frame_dir, self.cfg.gs_mode,"stream",f"iteration_{self.cfg.iter}", "stream_500",next_frame)
        # results["stream_path"] = stream_path


        # results["points"] = gs.get_xyz
        # resize render ground-truth images, range still in [0, 1]
        results['cur_images_input'] = multi_cur_images_input # [2,V, C, output_size, output_size]     
        results['next_images_input'] = multi_next_images_input # [2,V, C, output_size, output_size]      

        results['images_output'] = multi_next_images # [V, C, output_size, output_size]

        results["depth"] = depth_images_input
        # print(results["depths"].shape, results["depths"].dtype, results["depths"].max(), results["depths"].min())
        # results['images_input'] = images_input # [2,V, C, output_size, output_size]      
        # results['images_output'] = F.interpolate(next_images, size=(self.cfg.output_height, self.cfg.output_width), mode='bilinear', align_corners=False) # [V, C, output_size, output_size]
        
        #在这里就要是 3dgs/colmap的相机位姿
        results['c2w_output'] = c2ws
        results['c2w_input'] = c2ws_input
        results['FOV'] = torch.tensor([FovX, FovY], dtype=torch.float32)
        results["background_color"] = self.background_color

        results["resolution"] = torch.tensor([self.cfg.output_height, self.cfg.output_width])
        results["idx"] = idx

        results["radius"] = radius
        results["translate"] = translate

        # results["flows"] = flows
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
            results["local_rays"] = directions #local dir，这里暂时只用local dir

            #加上c2w
            # print(c2ws_input.shape, directions.shape)
            dirs = c2ws_input[:,:3,:3]@ directions.view(-1,3).permute(1,0).unsqueeze(0)
            # dirs = rearrange(dirs, " B D (H W) -> B H W D",H=H)

            ori = c2ws_input[:,:3,3].unsqueeze(-1).repeat_interleave(int(H*W), dim=-1)

            rays = torch.cat([ori, dirs], dim=1)
            rays = rearrange(rays, " B D (H W) -> B H W D",H=H)
            results["rays"] = rays
            # print(dirs.shape)

        return results

    def collate(self, batch):
        batch = torch.utils.data.default_collate(batch)
        # print("collate",batch['images_output'].device)
        gs_list = []
        bounding_box_list = []
        points_list = []

        for gs_path in batch["gs_path"]:
            gs = load_ply(gs_path)

            # bounding_box = gs.get_bounding_box
            bounding_box = torch.tensor([[-13,-1.5,7],[8,10,17]]) # for n3d
            # points = gs.get_xyz

            gs_list.append(gs)
            # points_list.append(points)
            bounding_box_list.append(bounding_box)
        bounding_box = torch.stack(bounding_box_list, dim=0)
        batch.update({"gs": gs_list, "bounding_box": bounding_box})

        if self.cfg.use_gstream:
            stream_masks = []
            stream_dxyzs = []
            stream_drots = []
            for stream_path in batch["stream_path"]:
                mask = torch.from_numpy(np.load(os.path.join(stream_path, "mask.npy")))
                d_xyz = torch.from_numpy(np.load(os.path.join(stream_path, "d_xyz.npy")))
                d_rot = torch.from_numpy(np.load(os.path.join(stream_path, "d_rot.npy")))
                arr = np.load(os.path.join(stream_path, "d_rot.npy"))
                assert not np.isnan(arr).any()
                stream_masks.append(mask)
                stream_dxyzs.append(d_xyz)
                stream_drots.append(d_rot)

            stream_masks = torch.cat(stream_masks, dim=0)
            stream_dxyzs = torch.cat(stream_dxyzs, dim=0)
            stream_drots = torch.cat(stream_drots, dim=0)
            batch.update({"stream_masks": stream_masks, "stream_dxyzs": stream_dxyzs, "stream_drots": stream_drots})

        return batch

