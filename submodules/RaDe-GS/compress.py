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
#Compress for the optimizaed Gaussians

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from utils.graphics_utils import point_double_to_normal, depth_double_to_normal

from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, OptimizationParams
from gaussian_renderer import GaussianModel
import cv2
import numpy as np
# from icecream import ic
from prune import prune_list, calculate_v_imp_score
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from random import randint

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, kernel_size):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_median_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth_median_mm")
    depth_expected_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth_expected_mm")

    alpha_path = os.path.join(model_path, name, "ours_{}".format(iteration), "alpha")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_median_path, exist_ok=True)
    makedirs(depth_expected_path, exist_ok=True)

    makedirs(alpha_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_pkg = render(view, gaussians, pipeline, background, kernel_size=kernel_size)
        print(kernel_size)
        rendering = render_pkg["render"]
        depth_median = render_pkg["median_depth"]
        depth_expected = render_pkg["expected_depth"]
        gt = view.original_image[0:3, :, :]

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        
        depth_median = (depth_median*1000).detach().cpu().squeeze().numpy().astype(np.uint16)
        depth_expected = (depth_expected*1000).detach().cpu().squeeze().numpy().astype(np.uint16)
        cv2.imwrite(os.path.join(depth_median_path, '{0:05d}'.format(idx) + ".png"), depth_median)
        cv2.imwrite(os.path.join(depth_expected_path, '{0:05d}'.format(idx) + ".png"), depth_expected)

def compress(dataset, opt, pipe, scene, iteration, gaussians, background, kernel_size):
    tb_writer = None
    training_report(tb_writer, iteration,None,None,None,l1_loss,None,args.test_iterations,scene, render, (pipe, background, kernel_size))
    i=0
    gaussian_list, imp_list = prune_list(gaussians, scene, pipe, background)

    if args.prune_type == "important_score":
        gaussians.prune_gaussians(
            (args.prune_decay**i) * args.prune_percent, imp_list
        )
    elif args.prune_type == "v_important_score":
        v_list = calculate_v_imp_score(gaussians, imp_list, args.v_pow)
        gaussians.prune_gaussians(
            (args.prune_decay**i) * args.prune_percent, v_list
        )
    elif args.prune_type == "max_v_important_score":
        v_list = imp_list * torch.max(gaussians.get_scaling, dim=1)[0]
        gaussians.prune_gaussians(
            (args.prune_decay**i) * args.prune_percent, v_list
        )
    elif args.prune_type == "count":
        gaussians.prune_gaussians(
            (args.prune_decay**i) * args.prune_percent, gaussian_list
        )
    elif args.prune_type == "opacity":
        gaussians.prune_gaussians(
            (args.prune_decay**i) * args.prune_percent,
            gaussians.get_opacity.detach(),
        )
    gaussians.compute_3D_filter(cameras=scene.getTrainCameras().copy())

    print("After prune iteration, number of gaussians: " + str(len(gaussians.get_xyz)))
    training_report(tb_writer, iteration,None,None,None,l1_loss,None,args.test_iterations,scene, render, (pipe, background, kernel_size))

    first_iter = iteration
    #接下来再迭代 1k-5k个epoch
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    trainCameras = scene.getTrainCameras().copy()
    # if dataset.disable_filter3D:
    #     gaussians.reset_3D_filter()
    # else:
    #     gaussians.compute_3D_filter(cameras=trainCameras)

    viewpoint_stack = None
    ema_loss_for_log, ema_depth_loss_for_log, ema_mask_loss_for_log, ema_normal_loss_for_log = 0.0, 0.0, 0.0, 0.0

    require_depth = not dataset.use_coord_map
    require_coord = dataset.use_coord_map
    
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam: Camera = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        # if (iteration - 1) == debug_from:
        #     pipe.debug = True

        reg_kick_on = iteration >= opt.regularization_from_iter
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, kernel_size, require_coord = require_coord and reg_kick_on, require_depth = require_depth and reg_kick_on)
        rendered_image: torch.Tensor
        rendered_image, viewspace_point_tensor, visibility_filter, radii = (
                                                                    render_pkg["render"], 
                                                                    render_pkg["viewspace_points"], 
                                                                    render_pkg["visibility_filter"], 
                                                                    render_pkg["radii"])
        gt_image = viewpoint_cam.original_image.cuda()

        if dataset.use_decoupled_appearance:
            Ll1_render = L1_loss_appearance(rendered_image, gt_image, gaussians, viewpoint_cam.uid)
        else:
            Ll1_render = l1_loss(rendered_image, gt_image)

        
        if reg_kick_on:
            lambda_depth_normal = opt.lambda_depth_normal
            if require_depth:
                rendered_expected_depth: torch.Tensor = render_pkg["expected_depth"]
                rendered_median_depth: torch.Tensor = render_pkg["median_depth"]
                rendered_normal: torch.Tensor = render_pkg["normal"]
                depth_middepth_normal = depth_double_to_normal(viewpoint_cam, rendered_expected_depth, rendered_median_depth)
            else:
                rendered_expected_coord: torch.Tensor = render_pkg["expected_coord"]
                rendered_median_coord: torch.Tensor = render_pkg["median_coord"]
                rendered_normal: torch.Tensor = render_pkg["normal"]
                depth_middepth_normal = point_double_to_normal(viewpoint_cam, rendered_expected_coord, rendered_median_coord)
            depth_ratio = 0.6
            normal_error_map = (1 - (rendered_normal.unsqueeze(0) * depth_middepth_normal).sum(dim=1))
            depth_normal_loss = (1-depth_ratio) * normal_error_map[0].mean() + depth_ratio * normal_error_map[1].mean()
        else:
            lambda_depth_normal = 0
            depth_normal_loss = torch.tensor([0],dtype=torch.float32,device="cuda")
            
        rgb_loss = (1.0 - opt.lambda_dssim) * Ll1_render + opt.lambda_dssim * (1.0 - ssim(rendered_image, gt_image.unsqueeze(0)))
        
        loss = rgb_loss + depth_normal_loss * lambda_depth_normal
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_normal_loss_for_log = 0.4 * depth_normal_loss.item() + 0.6 * ema_normal_loss_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{4}f}", "loss_normal": f"{ema_normal_loss_for_log:.{4}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
            
            # Log and save
            training_report(tb_writer, iteration, Ll1_render, loss, depth_normal_loss, l1_loss, iter_start.elapsed_time(iter_end), args.test_iterations, scene, render, (pipe, background, kernel_size))
        
            if (iteration in args.save_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                # scene.save(str(iteration)+"_compress_v2")
                scene.save(str(iteration)+"_compress")


                
            if iteration % 100 == 0 :
            # and iteration > opt.densify_until_iter and not dataset.disable_filter3D:
                # if iteration < opt.iterations - 100:
                    # don't update in the end of training
                    gaussians.compute_3D_filter(cameras=trainCameras)


            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            # if (iteration in checkpoint_iterations):
            #     print("\n[ITER {}] Saving Checkpoint".format(iteration))
            #     torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")



def training_report(tb_writer, iteration, Ll1, loss, normal_loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/normal_loss', normal_loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : scene.getTrainCameras()})
                            #   [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_result = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_result["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.cuda(), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if config["name"] == "test":
                    with open(scene.model_path + "/chkpnt_compress" + str(iteration) + ".txt", "w") as file_object:
                        print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test), file=file_object)
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

def render_sets(dataset : ModelParams, opt, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    # with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        gaussians.training_setup(opt)
        gaussians.max_radii2D = torch.zeros((gaussians.get_xyz.shape[0]), device="cuda")

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        compress(dataset, opt, pipeline, scene,  scene.loaded_iter, gaussians,  background, dataset.kernel_size)
        # render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.kernel_size)
        #再去做渲染
        
        # if not skip_train:
        #      render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.kernel_size)

        # if not skip_test:
        #      render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.kernel_size)

if __name__ == "__main__":
    #在这里定义的，是模块级别的变量
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser, sentinel=True) #要加sential，否则用不了cfg里面的参数
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--v_pow", type=float, default=0.1)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7000,10_000,11000,12000,13000,14000,15000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 10000,11000,12000,13000,14000,15000])

    parser.add_argument(
        "--prune_type", type=str, default="v_important_score"
    )  # k_mean, farther_point_sample, important_score

    parser.add_argument("--prune_percent", type=float, default=0.66)
    parser.add_argument("--prune_decay", type=float, default=1)

    args = get_combined_args(parser) #结合cfg和command line
    print("rade Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    # print(args.)
    args.test_iterations+=[i for i in range(4000,8000,100)]
    args.save_iterations+=[i for i in range(4000,8000,100)]
    # args.test_iterations+=[i for i in range(7000,10000,1000)]
    # args.save_iterations+=[i for i in range(7000,10000,1000)]
    render_sets(lp.extract(args),op.extract(args),  args.iteration, pp.extract(args), args.skip_train, args.skip_test)

