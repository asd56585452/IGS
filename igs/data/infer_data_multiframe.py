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

    # key_frame:bool = False
class N3dDataset(Dataset):

    # def _warn(self):
    #     raise NotImplementedError('this dataset is just an example and cannot be used directly, you should modify it to your own setting! (search keyword TODO)')
    '''
    除了第一帧，其余不需要load gaussian 和 input img
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
        # self.refine_items = [i for i in range(5,300,5)]
        self.refine_items = [i for i in range(5,300,5)]

        # self.refine_items = [1]
        print(self.refine_items)

        bbox_path = os.path.join(cfg.root_dir, self.cfg.bbox_path)
        with open(bbox_path, 'r') as f:
            bbox_path = json.load(f)
            self.bboxs = bbox_path

        cameras_json_path = os.path.join(os.path.join(self.cfg.root_dir, self.items[0]["scene_name"], self.items[0]["cur_frame"]),self.cfg.gs_mode,"cameras.json")
        with open(cameras_json_path) as f:
            self.cameras_data = json.load(f)
    def build_refine_dataset(self):
        refine_dataset = {}
        
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_item, idx, self) for idx in self.refine_items]
            
            for future in tqdm(as_completed(futures), total=len(futures)):
                idx, res_dict = future.result()
                refine_dataset.update({idx: res_dict})
        
        refine_dataset["resolution"] = torch.tensor([self.cfg.output_height, self.cfg.output_width])
        self.refine_dataset = refine_dataset
    # def build_refine_dataset(self):
    #     refine_dataset = {}
    #     for idx in tqdm(self.refine_items):
    #         scene_name = self.items[idx]["scene_name"]
    #         cur_frame = self.items[idx]["cur_frame"]
    #         cur_frame_dir = os.path.join(self.cfg.root_dir, scene_name, cur_frame)
    #         cameras_json_path = os.path.join(cur_frame_dir,self.cfg.gs_mode,"cameras.json")
    #         with open(cameras_json_path) as f:
    #             cameras_data = json.load(f)
    #         cur_images = []
    #         next_images = []
    #         depth_images = []
    #         c2ws = []
    #         results = {}
    #         # print(cameras_data)
    #         # exit()
    #         for camera  in cameras_data[1:]:
    #             image_name = camera["img_name"]
    #             # print(image_name, camera['id'])
    #             image_name_id = str(camera['id']).zfill(5) #render后用的是id命名的
    #             cur_image_path = os.path.join(cur_frame_dir, self.cfg.gs_mode,"train",f"ours_{self.cfg.iter}","renders", image_name_id+".png")
    #             #暂时不进行size变换
    #             cur_image = torch.from_numpy(np.array(Image.open(cur_image_path))/255.0).permute(2,0,1).to(torch.float)

    #             c2w = np.zeros((4, 4))
    #             c2w[:3,:3] = np.array(camera["rotation"])
    #             c2w[:3,3] = np.array(camera["position"])
    #             c2w[3,3] = 1
    #             c2w = torch.from_numpy(c2w).to(torch.float)

    #             fx = camera["fx"]
    #             fy =camera["fy"]
    #             width = camera["width"]
    #             height = camera["height"]

    #             FovX = focal2fov(fx, width)
    #             FovY = focal2fov(fy, height)

    #             cur_images.append(cur_image)
    #             c2ws.append(c2w)
    #         # cur_images = torch.stack(cur_images, dim=0) # [V, C, H, W]
    #         # c2ws = torch.stack(c2ws, dim=0) # [V, 4, 4]    
    #         FOV = torch.tensor([FovX, FovY], dtype=torch.float32)
    #         bg_color = self.background_color
    #         res_dict = {"images": cur_images,"c2ws":c2ws,"FOV":FOV,"bg":bg_color}

    #         refine_dataset.update({idx: res_dict})
    #     refine_dataset["resolution"] = torch.tensor([self.cfg.output_height, self.cfg.output_width])
    #     # refine_dataset["ba"]
    #     self.refine_dataset = refine_dataset
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
        # cur frame是一个 next frame是一堆。这个函数没有写完
        scene_name = self.items[idx]["scene_name"]
        cur_frame = self.items[idx]["cur_frame"]
        next_frame_list = self.items[idx]["next_frame"]
        keyframe = self.items[idx].get("keyframe", None)
        # ic(idx, cur_frame, next_frame)

        need_depth = False
        if cur_frame == "colmap_0":
            need_depth = True            

        # cur_frame = "colmap_70"
        # next_frame = "colmap_69"

        cur_frame_dir = os.path.join(self.cfg.root_dir, scene_name, cur_frame)

        # cameras_json_path = os.path.join(cur_frame_dir,self.cfg.gs_mode,"cameras.json")
        # with open(cameras_json_path) as f:
        #     cameras_data = json.load(f)
        cameras_data = self.cameras_data

        #读取radius和center
        cam_centers = []
        for cam in cameras_data:
            cam_centers.append(np.array(cam["position"])[...,np.newaxis])
        scene_info = getNerfppNorm(cam_centers)

        translate = scene_info["translate"]
        radius = scene_info["radius"]
        # group_json_path = os.path.join(self.cfg.root_dir, scene_name, "group.json")
        # with open(group_json_path) as f:
        #     group_data = json.load(f)

        # vids = np.arange(len(cameras_data)).tolist()

        # if self.cfg.scene_type =="n3d":
        #     eval_vids = [0]
        #     input_vids = list(set(vids) - set(eval_vids))

        #     eval_vids = torch.tensor(eval_vids)
        #     input_vids = torch.tensor(input_vids)
        #     # input_vids = torch.tensor(vids)

        bbox = torch.tensor(self.bboxs[scene_name]).to(torch.float)

        if self.cfg.scene_type =="n3d":
            eval_vids = [0]
            input_vids = [13, 1, 8, 4]

            vids =  eval_vids + input_vids
        elif self.cfg.scene_type == "meet":
            eval_vids = [0]
            input_vids = [3, 10, 1, 4]
            vids =  eval_vids + input_vids
        
        multi_cur_images = []
        multi_next_images = []
        multi_depth_images_input = []
        multi_c2ws = []
        multi_c2ws_input = []

        results = {}
        for tid in range(len(next_frame_list)):
            next_frame_dir = os.path.join(self.cfg.root_dir, scene_name, next_frame_list[tid])

            cur_images = []
            next_images = []
            depth_images = []
            c2ws = []
            results = {}
            for vid in vids:

                if self.cfg.scene_type =="n3d":
                    image_name_id = str(vid).zfill(5) #render后用的是id命名的

                    cur_image_path = os.path.join(cur_frame_dir, self.cfg.gs_mode,"train",f"ours_{self.cfg.iter}","gt", image_name_id+".png")
                    next_image_path = os.path.join(next_frame_dir, self.cfg.gs_mode,"train",f"ours_{self.cfg.iter}","gt", image_name_id+".png")
                    # next_image_path = os.path.join(next_frame_dir, "images", image_name+".png")
                    depth_image_path = os.path.join(cur_frame_dir, self.cfg.gs_mode,"train",f"ours_{self.cfg.iter}","depth_expected_mm", image_name_id+".png")

                elif self.cfg.scene_type == "meet":
                    image_name = cameras_data[vid]["img_name"]
                    image_name_id = str(vid).zfill(5) #render后用的是id命名的

                    cur_image_path = os.path.join(cur_frame_dir, "images", image_name+".png")
                    next_image_path = os.path.join(next_frame_dir,  "images", image_name+".png")
                    # next_image_path = os.path.join(next_frame_dir, "images", image_name+".png")
                    depth_image_path = os.path.join(cur_frame_dir, self.cfg.gs_mode,"train",f"ours_{self.cfg.iter}","depth_expected_mm", image_name_id+".png")

                


                #暂时不进行size变换
                cur_image = torch.from_numpy(np.array(Image.open(cur_image_path))/255.0).permute(2,0,1).to(torch.float)
                # print(cur_image.shape, vid, cur_frame, next_frame,)
                next_image = torch.from_numpy(np.array(Image.open(next_image_path))/255.0).permute(2,0,1).to(torch.float)
                
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
                c2ws.append(c2w)
            cur_images = torch.stack(cur_images, dim=0) # [V, C, H, W]
            next_images = torch.stack(next_images, dim=0) # [V,C, H, W]

            if need_depth:
                depth_images = torch.stack(depth_images, dim=0)
                depth_images_input = depth_images[1:].clone()
                multi_depth_images_input.append(depth_images_input)
            c2ws = torch.stack(c2ws, dim=0) # [V, 4, 4]
            c2ws_input = c2ws[1:].clone()
            multi_c2ws.append(c2ws)
            multi_c2ws_input.append(c2ws_input)


            cur_images_input = cur_images[1:].clone()

            next_images_input = next_images[1:].clone()

            multi_cur_images.append(cur_images)
            multi_next_images.append(next_images)
            multi_cur_images_input.append(cur_images_input)
            multi_next_images_input.append(next_images_input)

        results["depth"] = depth_images_input

        if idx ==0:
            # gs_path = os.path.join(cur_frame_dir, self.cfg.gs_mode+"_eval_purez","point_cloud",f"iteration_{self.cfg.iter}","point_cloud.ply")
            # gs_path = os.path.join(cur_frame_dir, self.cfg.gs_mode+"_eval_purez","point_cloud",f"iteration_7000","point_cloud.ply")
            # gs_path = "/workspace/40021/yjb/datasets/neural_3D/cut_roasted_beef_colmap/colmap_0/test/point_cloud/iteration_4900/point_cloud.ply"
            gs_path = self.cfg.start_gs_path
            # gs_path = "/workspace/40021/yjb/datasets/meeting_room/MeetRoom/discussion/colmap_64/3dgs_rade/point_cloud/iteration_7000/point_cloud.ply"
            results["gs_path"] = gs_path


        # next_images_input = torch.index_select(next_images.clone(), 0, input_vids)
        # c2ws_input = torch.index_select(c2ws.clone(), 0, input_vids)


        #load Gaussian
        # gs = GaussianModel()
        # print("load gs")
        # gs = load_ply(gs_path)
        # # print("ok")
        # bounding_box = gs.get_bounding_box
        # print(bounding_box.requires_grad)
        # results["bounding_box"] = bounding_box
        # results["gs"] = gs

        # results["points"] = gs.get_xyz
        # resize render ground-truth images, range still in [0, 1]
        if cur_images_input != None:
            results['cur_images_input'] = cur_images_input # [2,V, C, output_size, output_size]     

        results['next_images_input'] = next_images_input # [2,V, C, output_size, output_size]      

        results['images_output'] = next_images # [V, C, output_size, output_size]

        # print(results["depths"].shape, results["depths"].dtype, results["depths"].max(), results["depths"].min())
        # results['images_input'] = images_input # [2,V, C, output_size, output_size]      
        # results['images_output'] = F.interpolate(next_images, size=(self.cfg.output_height, self.cfg.output_width), mode='bilinear', align_corners=False) # [V, C, output_size, output_size]
        
        #在这里就要是 3dgs/colmap的相机位姿
        results['c2w_output'] = c2ws #只需要输出第一个就行
        results['c2w_input'] = c2ws_input
        # print(results['c2w_input'])
        # print(results['c2w_input'].shape, results['c2w_output'].shape, results['images_output'].shape, results['next_images_input'].shape, results['cur_images_input'].shape, results["depth"].shape)

        results['FOV'] = torch.tensor([FovX, FovY], dtype=torch.float32)
        results["background_color"] = self.background_color

        # results["resolution"] = torch.tensor([self.cfg.output_height, self.cfg.output_width])
        output_height, output_width = next_images_input.shape[-2:]
        results["resolution"] = torch.tensor([output_height, output_width])

        results["idx"] = idx
        results["eval_vids"] = eval_vids
        results["radius"] = radius
        results["bounding_box"] = bbox

        if keyframe !=None:
            results["keyframe"] = keyframe
        # if self.cfg.need_rays:
        #     H = self.cfg.input_height / 8
        #     W = self.cfg.input_width / 8
        #     fx , fy = fov2focal(FovX, W), fov2focal(FovY, H) 
        #     i, j = torch.meshgrid(
        #         torch.arange(W, dtype=torch.float32) + 0.5,
        #         torch.arange(H, dtype=torch.float32) + 0.5,
        #         indexing="xy",
        #     )

        #     directions: Float[Tensor, "H W 3"] = torch.stack(
        #         [(i - W/2) / fx, (j - H/2) / fy, torch.ones_like(i)], -1
        #     )
        #     directions = F.normalize(directions, p=2.0, dim=-1)
        #     results["rays"] = directions #local dir，这里暂时只用local dir

        if self.cfg.need_rays:
            H = int(self.cfg.input_height / 8)
            W = int(self.cfg.input_width / 8)
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
        # bounding_box_list = []
        points_list = []

        if "gs_path" not in batch:
            # only support batchsize =1
            return batch
        for gs_path in batch["gs_path"]:
            gs = load_ply(gs_path, self.cfg.max_sh_degree)
            # print(gs.get_xyz.shape)
            # np.save("trimming.npy", gs.get_xyz.cpu().detach().numpy())
            # # # storage = torch.cat([gs.get_xyz, gs.get_rotation, gs.get_opacity, gs.get_scaling],dim=-1).cpu().detach().numpy()
            # # # np.save("sorage.npy", storage)
            # gs.save_ply("./point_cloud.ply")
            # exit()
            # ic(gs)
            # bounding_box = gs.get_bounding_box
            # bounding_box = torch.tensor([[-13,-1.5,7],[8,10,17]])

            points = gs.get_xyz
            # np.save("points_spinach.npy", points.detach().cpu().numpy())
            # exit()
            gs_list.append(gs)
            points_list.append(points)
            # bounding_box_list.append(bounding_box)
        # bounding_box = torch.stack(bounding_box_list, dim=0)
        batch.update({"gs": gs_list, "points": points_list})
        return batch

    # def collate(self, batch):
    #     batch = torch.utils.data.default_collate(batch)
    #     # print("collate",batch['images_output'].device)
    #     gs_list = []
    #     bounding_box_list = []
    #     points_list = []
    #     for gs_path in batch["gs_path"]:
    #         gs = load_ply(gs_path)
    #         # bounding_box = gs.get_bounding_box
    #         bounding_box = torch.tensor([[-13,-1.5,7],[8,10,17]]) # for n3d
    #         points = gs.get_xyz

    #         gs_list.append(gs)
    #         points_list.append(points)
    #         bounding_box_list.append(bounding_box)
    #     bounding_box = torch.stack(bounding_box_list, dim=0)
    #     batch.update({"gs": gs_list, "bounding_box": bounding_box, "points": points_list})
    #     return batch

def process_item( idx, self):
    scene_name = self.items[idx-1]["scene_name"]
    cur_frame = self.items[idx-1]["next_frame"]# idx对应的是要被refine的帧，这个出现在idx-1的next_frame上
    cur_frame_dir = os.path.join(self.cfg.root_dir, scene_name, cur_frame)

    # cameras_json_path = os.path.join(cur_frame_dir, self.cfg.gs_mode, "cameras.json")
    # with open(cameras_json_path) as f:
    #     cameras_data = json.load(f)
    
    cameras_data = self.cameras_data

    cur_images = []
    c2ws = []
    # print(cameras_data)
    for camera in cameras_data[1:]:

        if self.cfg.scene_type =="n3d":
            image_name_id = str(camera['id']).zfill(5)
            cur_image_path = os.path.join(cur_frame_dir, self.cfg.gs_mode, "train", f"ours_{self.cfg.iter}", "renders", image_name_id + ".png")
        elif self.cfg.scene_type == "meet":
            image_name = camera["img_name"]
            cur_image_path = os.path.join(cur_frame_dir, "images", image_name + ".png")

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