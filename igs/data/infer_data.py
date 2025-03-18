import os
import cv2
import random
import json
from dataclasses import dataclass, field
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
from einops import rearrange

import kiui
from igs.models.gs import GaussianModel, load_ply
from icecream import ic
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

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

    start_frame:int = 0

    scene_type: Optional[str] = None
    need_rays:bool = True
    bbox_path: str = "bbox.json"
    start_gs_path: Optional[str] = None
    max_sh_degree:int = 3

    up_sample: bool =True
    # key_frame:bool = False
class N3dDataset(Dataset):

    '''
    only first frame need load gaussian
    '''
    def __init__(self, cfg:Any, training=True):
        super().__init__()

        self.cfg: N3dDatasetConfig = parse_structured(N3dDatasetConfig, cfg)

        # self.opt = opt
        self.training = training

        n3d_path = os.path.join(cfg.root_dir, self.cfg.data_path) # you can define your own split

        with open(n3d_path, 'r') as f:
            n3d_paths = json.load(f)
            if self.training:
                self.items = n3d_paths['train']
            else:
                self.items = n3d_paths['val']

       
        self.background_color = torch.as_tensor(self.cfg.background_color)
        self.refine_items = [i for i in range(5,300,5)]

        bbox_path = os.path.join(cfg.root_dir, self.cfg.bbox_path)
        with open(bbox_path, 'r') as f:
            bbox_path = json.load(f)
            self.bboxs = bbox_path

        cameras_json_path = os.path.join(os.path.join(self.cfg.root_dir, self.items[0]["scene_name"], self.items[0]["cur_frame"]),self.cfg.gs_mode,"cameras.json")
        with open(cameras_json_path) as f:
            self.cameras_data = json.load(f)

    def get_spiral(self, near=0.01,far=100, rads_scale=1.0, N_views=299):
        """
        Generate a set of poses using NeRF's spiral camera trajectory as validation poses.
        """
        # center pose
        c2ws_all = []
        for camera in self.cameras_data:

            c2w = np.zeros((3, 4))
            c2w[:3, :3] = np.array(camera["rotation"])
            c2w[:3,1:3] = -c2w[:3,1:3]

            c2w[:3, 3] = np.array(camera["position"])
            # c2w[3, 3] = 1
            c2ws_all.append(c2w)

            fx = camera["fx"]
        c2ws_all = np.stack(c2ws_all, axis=0)
        c2w = average_poses(c2ws_all)

        # Get average pose
        up = normalize(c2ws_all[:, :3, 1].sum(0))

        # Find a reasonable "focus depth" for this dataset
        dt = 0.75
        close_depth, inf_depth = near * 0.9, far* 5.0

        focal = 1.0 / ((1.0 - dt) / close_depth + dt / inf_depth)

        zdelta = near * 0.2
        tt = c2ws_all[:, :3, 3]
        rads = np.percentile(np.abs(tt), 90, 0) * rads_scale
        render_poses = render_path_spiral(
            c2w, up, rads, focal, zdelta, zrate=0.5, N=N_views
        )
        render_poses = np.stack(render_poses)
        self.free_poses = torch.from_numpy(render_poses).to(torch.float)
        return self.free_poses

    def build_refine_dataset(self,eval_batch_size):
        self.refine_items = [i for i in range(eval_batch_size,len(self.items)+1,eval_batch_size)]
        print("building key frames: ", self.refine_items)
        refine_dataset = {}
        
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_item, idx, self) for idx in self.refine_items]
            
            for future in tqdm(as_completed(futures), total=len(futures)):
                idx, res_dict = future.result()
                refine_dataset.update({idx: res_dict})
        
        # refine_dataset["resolution"] = torch.tensor([self.cfg.output_height, self.cfg.output_width])
        self.refine_dataset = refine_dataset

    def __len__(self):
        return len(self.items)
        

    def __getitem__(self, idx):
        '''
        {
            "input_cur_images":[V_input,H,W]
            "output_next_images":[V_output,H,W]
            "input_c2ws":[V_input,H,W]
            "output_c2ws":[V_output,H,W]
            "FOV"[2] 

            in one scene, FOV of cameras are smae
        }
        '''
        scene_name = self.items[idx]["scene_name"]
        cur_frame = self.items[idx]["cur_frame"]
        next_frame = self.items[idx]["next_frame"]
        keyframe = self.items[idx].get("keyframe", None)

        need_depth = False
        if cur_frame == "colmap_0":
            need_depth = True            


        cur_frame_dir = os.path.join(self.cfg.root_dir, scene_name, cur_frame)
        next_frame_dir = os.path.join(self.cfg.root_dir, scene_name, next_frame)


        cameras_data = self.cameras_data

        cam_centers = []
        for cam in cameras_data:
            cam_centers.append(np.array(cam["position"])[...,np.newaxis])
        scene_info = getNerfppNorm(cam_centers)

        translate = scene_info["translate"]
        radius = scene_info["radius"]


        bbox = torch.tensor(self.bboxs[scene_name]).to(torch.float)

        if self.cfg.scene_type =="n3d":
            eval_vids = [0]
            input_vids = [13, 1, 8, 4] # only 4 views

            vids =  eval_vids + input_vids
        elif self.cfg.scene_type == "meet":
            eval_vids = [0]
            input_vids = [3, 10, 1, 4]
            vids =  eval_vids + input_vids
        elif self.cfg.scene_type == "enerf":
            eval_vids = [0]
            input_vids = [9, 2, 3, 1]
            vids =  eval_vids + input_vids
        cur_images = []
        next_images = []
        cur_images_resize = []
        next_images_resize = []

        depth_images = []
        c2ws = []
        results = {}
        for vid in vids:
            if self.cfg.scene_type =="n3d":
                image_name_id = str(vid+1).zfill(5) #need plus 1 here to get the start depth
                image_name = cameras_data[vid]["img_name"]

                depth_image_path = os.path.join(cur_frame_dir, self.cfg.gs_mode,"train",f"ours_{self.cfg.iter}","depth_expected_mm", image_name_id+".png")
                cur_image_path = os.path.join(cur_frame_dir, "images_r2", image_name+".png")
                next_image_path = os.path.join(next_frame_dir,  "images_r2", image_name+".png")
                
                cur_image_path_resize = os.path.join(cur_frame_dir, "images_512", image_name+".png")
                next_image_path_resize = os.path.join(next_frame_dir,  "images_512", image_name+".png")
            elif self.cfg.scene_type == "meet":
                image_name = cameras_data[vid]["img_name"]
                image_name_id = str(vid+1).zfill(5) 

                cur_image_path = os.path.join(cur_frame_dir, "images", image_name+".png")
                next_image_path = os.path.join(next_frame_dir,  "images", image_name+".png")
                cur_image_path_resize = os.path.join(cur_frame_dir, "images_512", image_name+".png")
                next_image_path_resize = os.path.join(next_frame_dir,  "images_512", image_name+".png")
     
                depth_image_path = os.path.join(cur_frame_dir, self.cfg.gs_mode,"train",f"ours_{self.cfg.iter}","depth_expected_mm", image_name_id+".png")

            elif self.cfg.scene_type == "enerf":

                image_name = cameras_data[vid]["img_name"]
                depth_vid = vid-1
                if depth_vid == -1:
                    depth_vid = 0
                image_name_id = str(depth_vid).zfill(5) #render后用的是id命名的

                cur_image_path = os.path.join(cur_frame_dir, "images_2", image_name+".jpg")
                next_image_path = os.path.join(next_frame_dir,  "images_2", image_name+".jpg")
            
                cur_image_path_resize = os.path.join(cur_frame_dir, "images_512", image_name+".jpg")
                next_image_path_resize = os.path.join(next_frame_dir,  "images_512", image_name+".jpg")
     
                depth_image_path = os.path.join(cur_frame_dir, self.cfg.gs_mode,"train",f"ours_{self.cfg.iter}","depth_expected_mm", image_name_id+".png")

            


            #暂时不进行size变换
            cur_image = torch.from_numpy(np.array(Image.open(cur_image_path))/255.0).permute(2,0,1).to(torch.float)
            next_image = torch.from_numpy(np.array(Image.open(next_image_path))/255.0).permute(2,0,1).to(torch.float)
            cur_image_resize = torch.from_numpy(np.array(Image.open(cur_image_path_resize))/255.0).permute(2,0,1).to(torch.float)
            next_image_resize = torch.from_numpy(np.array(Image.open(next_image_path_resize))/255.0).permute(2,0,1).to(torch.float)
                   
            if need_depth:
                depth_image = torch.from_numpy(np.array(Image.open(depth_image_path))/1000.0).to(torch.float)
                depth_images.append(depth_image)




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
            cur_images_resize.append(cur_image_resize)
            next_images_resize.append(next_image_resize)
            c2ws.append(c2w)
        cur_images = torch.stack(cur_images, dim=0) # [V, C, H, W]
        next_images = torch.stack(next_images, dim=0) # [V,C, H, W]
        cur_images_resize = torch.stack(cur_images_resize, dim=0) # [V, C, H, W]
        next_images_resize = torch.stack(next_images_resize, dim=0) # [V,C, H, W]   

        if need_depth:
            depth_images = torch.stack(depth_images, dim=0)

        c2ws = torch.stack(c2ws, dim=0) # [V, 4, 4]


        cur_images_input = None
        depth_images_input = None
        gs_path = None


        if need_depth:
            depth_images_input = depth_images[1:].clone()
            results["depth"] = depth_images_input

        if idx ==0:
            gs_path = self.cfg.start_gs_path
            results["gs_path"] = gs_path
        else:
            results["gs_path"] = ""

        cur_images_input = cur_images_resize[1:].clone()
        next_images_input = next_images_resize[1:].clone()
        c2ws_input = c2ws[1:].clone()

        if cur_images_input != None:
            results['cur_images_input'] = cur_images_input # [2,V, C, output_size, output_size]     

        results['next_images_input'] = next_images_input # [2,V, C, output_size, output_size]      

        results['images_output'] = next_images # [V, C, output_size, output_size]


        #在这里就要是 3dgs/colmap的相机位姿
        results['c2w_output'] = c2ws #只需要输出第一个就行
        results['c2w_input'] = c2ws_input

        results['FOV'] = torch.tensor([FovX, FovY], dtype=torch.float32)
        results["background_color"] = self.background_color

        output_height, output_width = next_images.shape[-2:]
        results["resolution"] = torch.tensor([output_height, output_width])

        results["idx"] = idx
        results["eval_vids"] = eval_vids
        results["radius"] = radius
        results["bounding_box"] = bbox

        if keyframe !=None:
            results["keyframe"] = keyframe

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
            results["local_rays"] = directions #local dir

            #take c2w

            dirs = c2ws_input[:,:3,:3]@ directions.view(-1,3).permute(1,0).unsqueeze(0)

            ori = c2ws_input[:,:3,3].unsqueeze(-1).repeat_interleave(int(H*W), dim=-1)

            rays = torch.cat([ori, dirs], dim=1)
            rays = rearrange(rays, " B D (H W) -> B H W D",H=H)
            results["rays"] = rays

        return results

    def collate(self, batch):
        batch = torch.utils.data.default_collate(batch)
        gs_list = []
        points_list = []

        if "gs_path" not in batch:
            return batch
        for gs_path in batch["gs_path"]:
            if gs_path != "":
                gs = load_ply(gs_path, self.cfg.max_sh_degree)
                points = gs.get_xyz
                gs_list.append(gs)
                points_list.append(points)

        batch.update({"gs": gs_list, "points": points_list})
        return batch


def process_item( idx, self):
    scene_name = self.items[idx-1]["scene_name"]
    cur_frame = self.items[idx-1]["next_frame"]# idx对应的是要被refine的帧，这个出现在idx-1的next_frame上
    cur_frame_dir = os.path.join(self.cfg.root_dir, scene_name, cur_frame)

    
    cameras_data = self.cameras_data

    cur_images = []
    c2ws = []

    cameras_data = cameras_data[1:] #only use traning views!
    for camera in cameras_data:

        if self.cfg.scene_type =="n3d":
            image_name = camera["img_name"]
            cur_image_path = os.path.join(cur_frame_dir, "images_r2", image_name + ".png")

        elif self.cfg.scene_type == "meet":
            image_name = camera["img_name"]
            cur_image_path = os.path.join(cur_frame_dir, "images", image_name + ".png")
        elif self.cfg.scene_type == "enerf":
            image_name = camera["img_name"]
            cur_image_path = os.path.join(cur_frame_dir, "images_2", image_name + ".jpg")
        cur_image = torch.from_numpy(np.array(Image.open(cur_image_path)) / 255.0).permute(2, 0, 1).to(torch.float)

        c2w = np.zeros((4, 4))
        c2w[:3, :3] = np.array(camera["rotation"])
        c2w[:3, 3] = np.array(camera["position"])
        c2w[3, 3] = 1
        c2w = torch.from_numpy(c2w).to(torch.float)

        fx = camera["fx"]
        fy = camera["fy"]
        width = camera["width"]
        height = camera["height"]

        FovX = focal2fov(fx, width)
        FovY = focal2fov(fy, height)

        cur_images.append(cur_image)
        c2ws.append(c2w)
    
    FOV = torch.tensor([FovX, FovY], dtype=torch.float32)
    bg_color = self.background_color
    res_dict = {"images": cur_images, "c2ws": c2ws, "FOV": FOV, "bg": bg_color}
    
    return idx, res_dict

def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, N_rots=2, N=120):
    render_poses = []
    rads = np.array(list(rads) + [1.0])

    focal = 18.35
    for theta in np.linspace(0.0, 2.0 * np.pi * N_rots, N + 1)[:-1]:
        c = np.dot(
            c2w[:3, :4],
            np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.0])
            * rads,
        )
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.0])))


        pose  = viewmatrix(z, up, c)
        pose_ = np.eye(4)
        pose_[:3,:] = pose[:3,:]
        R = pose_[:3,:3]
        R = - R
        T = -pose_[:3,3].dot(R)

        p = np.eye(4)
        p[:3,:3] = R.transpose()
        p[:3,3]=T
        p = np.linalg.inv(p)
        render_poses.append(p)


    return render_poses

def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.

    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)
    # 4. Compute the x axis
    x = normalize(np.cross(z,y_))  # (3)
    # x = normalize(np.cross(y_,z))  # (3)

    # 5. Compute te y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(x,z)  # (3)
    # y = np.cross(z,x)  # (3)

    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

    return pose_avg

def normalize(v):
    return v / np.linalg.norm(v)
def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.eye(4)
    m[:3] = np.stack([-vec0, vec1, vec2, pos], 1)
    # m[:3] = np.stack([vec0, vec1, vec2, pos], 1)

    return m