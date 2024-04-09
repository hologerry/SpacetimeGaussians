# MIT License

# Copyright (c) 2023 OPPO

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ========================================================================================================
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the thirdparty/gaussian_splatting/LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import json
import os
import pickle
import random
import sys
import time
import uuid

from argparse import ArgumentParser, Namespace
from random import randint

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from torchvision.utils import save_image
from tqdm import tqdm

from helper_train import (
    get_loss,
    get_model,
    get_render_pipe,
    reload_helper,
    remove_min_max,
    trb_function,
    undistort_image,
)
from thirdparty.gaussian_splatting.arguments import (
    ModelParams,
    OptimizationParams,
    PipelineParams,
)
from thirdparty.gaussian_splatting.scene import Scene
from thirdparty.gaussian_splatting.utils.general_utils import safe_state
from thirdparty.gaussian_splatting.utils.image_utils import psnr

### do no
from thirdparty.gaussian_splatting.utils.loss_utils import (
    l1_loss,
    l2_loss,
    relative_loss,
    ssim,
    ssim_map,
)


try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
EPS = 1e-6


def freeze_weights_by_mask_no_unsqueeze(model, screen_list, mask):
    for k in screen_list:
        grad_tensor = getattr(getattr(model, k), "grad")
        new_grad = mask * grad_tensor  # torch.zeros_like(grad_tensor)
        setattr(getattr(model, k), "grad", new_grad)
    return


def save_pkl(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def train(
    model_args,
    optim_args,
    pipe_args,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    densify=0,
    duration=50,
    basic_function="gaussian",
    rgb_function="rgbv1",
    rd_pipe="v2",
):
    first_iter = 0
    render, GRsetting, GRzer = get_render_pipe(rd_pipe)

    tb_writer = prepare_output_and_logger(model_args)
    print("use model {}".format(model_args.model))
    GaussianModel = get_model(model_args.model)

    gaussians = GaussianModel(model_args.sh_degree, rgb_function)
    gaussians.trbf_scale_init = -1 * optim_args.trbf_scale_init  # control the scale of trbf
    gaussians.preprocess_points = optim_args.preprocess_points

    trbf_base_function = trb_function
    scene = Scene(model_args, gaussians, duration=duration, loader=model_args.loader)

    gaussians.training_setup(optim_args)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, optim_args)
    num_channel = 9

    bg_color = [1, 1, 1] if model_args.white_background else [0 for i in range(num_channel)]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    # if freeze != 1:
    first_iter = 0
    progress_bar = tqdm(range(first_iter, optim_args.iterations), desc="Training progress")
    first_iter += 1

    flag = 0
    close_threshold = None
    depth_dict = {}

    if optim_args.batch > 1:
        train_camera_list = scene.getTrainCameras().copy()
        train_cam_dict = {}
        for i in range(duration):  # 0 to 4, -> (0.0, to 0.8)
            train_cam_dict[i] = [cam for cam in train_camera_list if cam.timestamp == i / duration]

    scale_threshold = gaussians.percent_dense * scene.cameras_extent

    if gaussians.ts is None:
        H, W = train_camera_list[0].image_height, train_camera_list[0].image_width
        gaussians.ts = torch.ones(1, 1, H, W).cuda()

    scene.record_points(0, "start training")
    start_time = time.time()
    gaussians.ray_start = optim_args.ray_start

    current_xyz = gaussians._xyz

    max_x, max_y, max_z = (
        torch.amax(current_xyz[:, 0]),
        torch.amax(current_xyz[:, 1]),
        torch.amax(current_xyz[:, 2]),
    )  #
    min_x, min_y, min_z = torch.amin(current_xyz[:, 0]), torch.amin(current_xyz[:, 1]), torch.amin(current_xyz[:, 2])

    if os.path.exists(optim_args.prev_path):  # reload trained model to boost results.
        print("load from " + optim_args.prev_path)
        reload_helper(gaussians, optim_args, max_x, max_y, max_z, min_x, min_y, min_z)

    max_bounds = [max_x, max_y, max_z]
    min_bounds = [min_x, min_y, min_z]

    flag_ems = 0  # change to 1 to start ems
    ems_cnt = 0
    max_loss = None
    max_loss_camera = None
    loss_dict = {}
    ssim_dict = {}
    depth_dict = {}
    valid_depth_dict = {}
    ems_start_from_iterations = optim_args.ems_start
    assert optim_args.loss_tart < optim_args.ems_start

    with torch.no_grad():
        time_index = 0  # 0 to 49
        view_point_set = train_cam_dict[time_index]

        for viewpoint_cam in view_point_set:
            render_pkg = render(
                viewpoint_cam,
                gaussians,
                pipe_args,
                background,
                override_color=None,
                basic_function=trbf_base_function,
                GRsetting=GRsetting,
                GRzer=GRzer,
            )

            _, depthH, depthW = render_pkg["depth"].shape
            border_H = int(depthH / 2)
            border_W = int(depthW / 2)

            mid_h = int(viewpoint_cam.image_height / 2)
            mid_w = int(viewpoint_cam.image_width / 2)

            depth = render_pkg["depth"]
            select_mask = depth != 15.0

            valid_depth_dict[viewpoint_cam.image_name] = depth[select_mask].var().item()
            depth_dict[viewpoint_cam.image_name] = torch.amax(depth[select_mask]).item()
            ssim_dict[viewpoint_cam.image_name] = ssim(
                render_pkg["render"].detach(), viewpoint_cam.original_image.float().detach()
            ).item()

    ordered_loss_dict = sorted(ssim_dict.items(), key=lambda item: item[1], reverse=False)
    ordered_depth = sorted(valid_depth_dict.items(), key=lambda item: item[1], reverse=True)

    total_length = len(ordered_depth)
    mid = int(total_length / 2)
    mid_depth_list = [p[0] for p in ordered_depth[:mid]]

    for k in mid_depth_list:
        scene.record_points(0, "selective: " + k)

    mid_loss_list = [p[0] for p in ordered_loss_dict[:mid]]

    dataset_root = os.path.dirname(model_args.source_path)
    picked_views_path = os.path.join(dataset_root, "pick_view.pkl")
    select_views = mid_loss_list[1:4]
    if not os.path.exists(picked_views_path):
        print("please copy pick view")
        quit()
        # with open(picked_views_path, 'wb') as handle:
        #     pickle.dump(select_views, handle, protocol=pickle.HIGHEST_PROTOCOL) # uncomment to dump the select_view to the model_args please select the duration = 1

    else:
        select_views = load_pkl(picked_views_path)

    for k in select_views:
        scene.record_points(0, "load: " + k)

    selected_length = 3
    laster_ems = 0
    last_rest = 0

    for iteration in range(first_iter, optim_args.iterations + 1):

        if iteration == optim_args.ems_start:
            flag_ems = 2  # start ems

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.one_up_sh_degree()

        # Pick a random Camera, or fewer than batch to pop
        if optim_args.batch == 1 and not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        if (iteration - 1) == debug_from:
            pipe_args.debug = True
        if gaussians.rgb_decoder is not None:
            gaussians.rgb_decoder.train()

        if optim_args.batch > 1:
            gaussians.zero_gradient_cache()

            time_index = randint(0, duration - 1)  # 0 to 49
            view_point_set = train_cam_dict[time_index]
            cam_index = random.sample(view_point_set, optim_args.batch)

            for i in range(optim_args.batch):
                viewpoint_cam = cam_index[i]
                render_pkg = render(
                    viewpoint_cam,
                    gaussians,
                    pipe_args,
                    background,
                    override_color=None,
                    basic_function=trbf_base_function,
                    GRsetting=GRsetting,
                    GRzer=GRzer,
                )
                image, viewspace_point_tensor, visibility_filter, radii = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                )
                gt_image = viewpoint_cam.original_image.float().cuda()

                if optim_args.reg == 2:
                    Ll1 = l2_loss(image, gt_image)
                    loss = Ll1
                elif optim_args.reg == 3:
                    Ll1 = relative_loss(image, gt_image)
                    loss = Ll1
                else:
                    Ll1 = l1_loss(image, gt_image)
                    loss = get_loss(optim_args, Ll1, ssim, image, gt_image, gaussians, radii)

                loss.backward()
                gaussians.cache_gradient()
                gaussians.optimizer.zero_grad(set_to_none=True)  #

            iter_end.record()
            gaussians.set_batch_gradient(optim_args.batch)

        else:
            raise NotImplementedError("Batch size 1 is not supported anymore")

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == optim_args.iterations:
                progress_bar.close()

            # Log and save                                                                                                         #viewpoint_camera, pc : GaussianModel, pipe_args, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, basic_function = None
            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                duration_time = time.time() - start_time
                txt_path = scene.model_path + "/training_time.txt"
                with open(txt_path, "w") as f:
                    f.write(str(iteration) + " cost time: " + str(duration_time))

            # ensure that parameters are same as in the model
            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/ckp" + str(iteration) + ".pth")

            # Densification

            if densify == 4:  #
                if iteration < optim_args.densify_until_iter:
                    gaussians.max_radii2D[visibility_filter] = torch.max(
                        gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                    )
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                    if iteration == 8001:  # 8001
                        omega_mask = gaussians.zero_omega_by_motion()  # 1 we keep omega, 0 we freeze omega
                        gaussians.omega_mask = omega_mask
                        scene.record_points(iteration, "separate omega" + str(torch.sum(omega_mask).item()))
                    elif iteration > 8001:  # 8001
                        freeze_weights_by_mask_no_unsqueeze(gaussians, ["_omega"], gaussians.omega_mask)
                        rotation_mask = torch.logical_not(gaussians.omega_mask)
                        freeze_weights_by_mask_no_unsqueeze(gaussians, ["_rotation"], rotation_mask)
                    if iteration > optim_args.densify_from_iter and iteration % optim_args.densification_interval == 0:
                        if flag < optim_args.densify_cnt:
                            scene.record_points(iteration, "before densify")
                            size_threshold = 20 if iteration > optim_args.opacity_reset_interval else None
                            gaussians.densify_prune_clone_im(
                                optim_args.densify_grad_threshold,
                                optim_args.opacity_threshold,
                                scene.cameras_extent,
                                size_threshold,
                            )
                            flag += 1
                            scene.record_points(iteration, "after densify")
                        else:
                            if iteration < 5000:
                                prune_mask = (gaussians.get_opacity < optim_args.opacity_threshold).squeeze()
                                if optim_args.prune_by_size:
                                    big_points_vs = gaussians.max_radii2D > 20
                                    big_points_ws = (
                                        gaussians.get_scaling.max(dim=1).values > 0.1 * scene.cameras_extent
                                    )
                                    prune_mask = torch.logical_or(
                                        torch.logical_or(prune_mask, big_points_vs), big_points_ws
                                    )
                                if (iteration > (500 + last_rest)) and last_rest > 1000:  #
                                    gaussians.prune_points(prune_mask)
                                    torch.cuda.empty_cache()
                                    scene.record_points(iteration, "additionally prune_mask")
                    if iteration % optim_args.opacity_reset_interval == 0 and iteration < 4000:
                        gaussians.reset_opacity()
                        last_rest = iteration

            if densify == 6:  #
                if iteration < optim_args.densify_until_iter:
                    gaussians.max_radii2D[visibility_filter] = torch.max(
                        gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                    )
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                    if iteration > optim_args.densify_from_iter and iteration % optim_args.densification_interval == 0:
                        if flag < optim_args.densify_cnt:
                            scene.record_points(iteration, "before densify")
                            size_threshold = 20 if iteration > optim_args.opacity_reset_interval else None
                            gaussians.densify_prune_clone_image_neral(
                                optim_args.densify_grad_threshold,
                                optim_args.opacity_threshold,
                                scene.cameras_extent,
                                size_threshold,
                            )
                            flag += 1
                            scene.record_points(iteration, "after densify")
                        else:
                            if iteration < 9000:
                                prune_mask = (gaussians.get_opacity < optim_args.opacity_threshold).squeeze()
                                if optim_args.prune_by_size:
                                    big_points_vs = gaussians.max_radii2D > 20
                                    big_points_ws = (
                                        gaussians.get_scaling.max(dim=1).values > 0.1 * scene.cameras_extent
                                    )
                                    prune_mask = torch.logical_or(
                                        torch.logical_or(prune_mask, big_points_vs), big_points_ws
                                    )
                                gaussians.prune_points(prune_mask)
                                torch.cuda.empty_cache()
                                scene.record_points(iteration, "additionally prune_mask")
                    if iteration % optim_args.opacity_reset_interval == 0:
                        gaussians.reset_opacity()

            if densify == 7:  # more general
                if iteration < optim_args.densify_until_iter:
                    gaussians.max_radii2D[visibility_filter] = torch.max(
                        gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                    )
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                    if iteration > optim_args.densify_from_iter and iteration % optim_args.densification_interval == 0:
                        if flag < optim_args.densify_cnt:
                            scene.record_points(iteration, "before densify")
                            size_threshold = 20 if iteration > optim_args.opacity_reset_interval else None
                            gaussians.densify_prune_clone_image_neral(
                                optim_args.densify_grad_threshold,
                                optim_args.opacity_threshold,
                                scene.cameras_extent,
                                size_threshold,
                            )
                            flag += 1
                            scene.record_points(iteration, "after densify")
                        else:
                            prune_mask = (gaussians.get_opacity < optim_args.opacity_threshold).squeeze()
                            if optim_args.prune_by_size:
                                big_points_vs = gaussians.max_radii2D > 20
                                big_points_ws = gaussians.get_scaling.max(dim=1).values > 0.1 * scene.cameras_extent
                                prune_mask = torch.logical_or(
                                    torch.logical_or(prune_mask, big_points_vs), big_points_ws
                                )
                            gaussians.prune_points(prune_mask)
                            torch.cuda.empty_cache()
                            scene.record_points(iteration, "additionally prune_mask")
                    if iteration % optim_args.opacity_reset_interval == 0:
                        gaussians.reset_opacity()

            if densify == 8:  # more generate method also remove min_max points
                if iteration < optim_args.densify_until_iter:
                    gaussians.max_radii2D[visibility_filter] = torch.max(
                        gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                    )
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                    if iteration > optim_args.densify_from_iter and iteration % optim_args.densification_interval == 0:
                        if flag < optim_args.densify_cnt:
                            scene.record_points(iteration, "before densify")
                            size_threshold = 20 if iteration > optim_args.opacity_reset_interval else None
                            gaussians.densify_prune_clone_image_neral(
                                optim_args.densify_grad_threshold,
                                optim_args.opacity_threshold,
                                scene.cameras_extent,
                                size_threshold,
                            )
                            flag += 1
                            scene.record_points(iteration, "after densify")
                        else:
                            prune_mask = (gaussians.get_opacity < optim_args.opacity_threshold).squeeze()
                            if optim_args.prune_by_size:
                                big_points_vs = gaussians.max_radii2D > 20
                                big_points_ws = gaussians.get_scaling.max(dim=1).values > 0.1 * scene.cameras_extent
                                prune_mask = torch.logical_or(
                                    torch.logical_or(prune_mask, big_points_vs), big_points_ws
                                )
                            gaussians.prune_points(prune_mask)
                            torch.cuda.empty_cache()
                            scene.record_points(iteration, "additionally prune_mask")
                    if iteration % optim_args.opacity_reset_interval == 0:
                        gaussians.reset_opacity()

                if iteration == 10000:
                    remove_min_max(gaussians, max_bounds, min_bounds)
                    torch.cuda.empty_cache()
                    scene.record_points(iteration, "additionally prune_mask")

            # after densification
            if (
                iteration > ems_start_from_iterations
                and flag_ems == 2
                and ems_cnt < selected_length
                and viewpoint_cam.image_name in select_views
                and (iteration - laster_ems > 100)
            ):  #
                scene.record_points(iteration, "current ems time " + str(time_index))

                select_views.remove(viewpoint_cam.image_name)

                ems_cnt += 1
                laster_ems = iteration

                diff = 1.0 - ssim_map(image.detach(), gt_image)  # we choose ares with large d-ssim..
                diff = torch.sum(diff, dim=0)  # h, w
                diff_sorted, _ = torch.sort(diff.reshape(-1))
                num_pixels = diff.shape[0] * diff.shape[1]
                threshold = diff_sorted[int(num_pixels * optim_args.ems_threshold)].item()

                out_mask = diff > threshold  # 0.03  #error threshold
                kh, kw = 16, 16  # kernel size
                dh, dw = 16, 16  # stride
                ideal_h, ideal_w = (
                    int(image.shape[1] / dh + 1) * kw,
                    int(image.shape[2] / dw + 1) * kw,
                )  # compute the ideal size for padding
                out_mask = F.pad(
                    out_mask,
                    (0, ideal_w - out_mask.shape[1], 0, ideal_h - out_mask.shape[0]),
                    mode="constant",
                    value=0,
                )

                patches = out_mask.unfold(0, kh, dh).unfold(1, kw, dw)
                dummy_patch = torch.ones_like(patches)
                patches_sum = patches.sum(dim=(2, 3))
                patches_mask = patches_sum > kh * kh * 0.85
                patches_mask = patches_mask.unsqueeze(2).unsqueeze(3).repeat(1, 1, kh, kh).float()
                patches = dummy_patch * patches_mask

                # mid_patch = torch.ones_like(patches)
                depth = render_pkg["depth"]
                depth = depth.squeeze(0)
                ideal_depth_h, ideal_depth_w = (
                    int(depth.shape[0] / dh + 1) * kw,
                    int(depth.shape[1] / dw + 1) * kw,
                )  # compute the ideal size for padding

                depth = torch.nn.functional.pad(
                    depth,
                    (0, ideal_depth_w - depth.shape[1], 0, ideal_depth_h - depth.shape[0]),
                    mode="constant",
                    value=0,
                )

                depth_patches = depth.unfold(0, kh, dh).unfold(1, kw, dw)
                dummy_depth_patches = torch.ones_like(depth_patches)
                a, b, c, d = depth_patches.shape
                depth_patches = depth_patches.reshape(a, b, c * d)
                median_depth_patch = torch.median(depth_patches, dim=(2))[0]
                depth_patches = dummy_depth_patches * (median_depth_patch.unsqueeze(2).unsqueeze(3))
                unfold_depth_shape = dummy_depth_patches.size()
                output_depth_h = unfold_depth_shape[0] * unfold_depth_shape[2]
                output_depth_w = unfold_depth_shape[1] * unfold_depth_shape[3]

                patches_depth_orig = depth_patches.view(unfold_depth_shape)
                patches_depth_orig = patches_depth_orig.permute(0, 2, 1, 3).contiguous()
                # H * W  mask, # 1 for error, 0 for no error
                patches_depth = patches_depth_orig.view(output_depth_h, output_depth_w).float()

                depth = patches_depth[: render_pkg["depth"].shape[1], : render_pkg["depth"].shape[2]]
                depth = depth.unsqueeze(0)

                mid_patch = torch.ones_like(patches)

                center_patches = patches * mid_patch

                unfold_shape = patches.size()
                patches_orig = patches.view(unfold_shape)
                center_patches_orig = center_patches.view(unfold_shape)

                output_h = unfold_shape[0] * unfold_shape[2]
                output_w = unfold_shape[1] * unfold_shape[3]
                patches_orig = patches_orig.permute(0, 2, 1, 3).contiguous()
                center_patches_orig = center_patches_orig.permute(0, 2, 1, 3).contiguous()
                # H * W  mask, # 1 for error, 0 for no error
                center_mask = center_patches_orig.view(output_h, output_w).float()
                center_mask = center_mask[: image.shape[1], : image.shape[2]]  # reverse back

                error_mask = patches_orig.view(
                    output_h, output_w
                ).float()  # H * W  mask, # 1 for error, 0 for no error
                error_mask = error_mask[: image.shape[1], : image.shape[2]]  # reverse back

                H, W = center_mask.shape

                offsetH = int(H / 10)
                offsetW = int(W / 10)  # fish eye boundary artifacts, we don't sample there

                center_mask[0:offsetH, :] = 0.0
                center_mask[:, 0:offsetW] = 0.0

                center_mask[-offsetH:, :] = 0.0
                center_mask[:, -offsetW:] = 0.0

                depth_map = torch.cat((depth, depth, depth), dim=0)
                invalid_depth_mask = depth == 15.0

                path_dir = scene.model_path + "/ems_" + str(ems_cnt - 1)
                if not os.path.exists(path_dir):
                    os.makedirs(path_dir)

                depth_map = depth_map / torch.amax(depth_map)
                invalid_depth_map = torch.cat(
                    (invalid_depth_mask, invalid_depth_mask, invalid_depth_mask), dim=0
                ).float()

                save_image(gt_image, os.path.join(path_dir, "gt" + str(iteration) + ".png"))
                save_image(image, os.path.join(path_dir, "render" + str(iteration) + ".png"))
                save_image(depth_map, os.path.join(path_dir, "depth" + str(iteration) + ".png"))
                save_image(invalid_depth_map, os.path.join(path_dir, "invalid_depth" + str(iteration) + ".png"))

                center_masked_images = torch.stack((center_mask, center_mask, center_mask), dim=2).float().cpu()  # 0,1
                center_masked_images = center_masked_images.numpy()
                # resize to x2
                # masked_images = cv2.resize()
                center_masked_images = cv2.resize(
                    center_masked_images,
                    dsize=(viewpoint_cam.image_width, viewpoint_cam.image_height),
                    interpolation=cv2.INTER_CUBIC,
                )

                # retrieve current camera's K

                ud_center_masked_images = undistort_image(
                    viewpoint_cam.image_name, model_args.source_path, center_masked_images
                )
                gt_image_numpy = gt_image.clone().permute(1, 2, 0).cpu().numpy()
                gt_image_x2 = cv2.resize(
                    gt_image_numpy,
                    dsize=(viewpoint_cam.image_width, viewpoint_cam.image_height),
                    interpolation=cv2.INTER_CUBIC,
                )
                gt_image_x2_ud = undistort_image(viewpoint_cam.image_name, model_args.source_path, gt_image_x2)
                gt_image_x2_ud_torch = torch.from_numpy(gt_image_x2_ud).cuda().permute(2, 0, 1)

                # use opencv undistort points to undistort these points
                ud_center_masked_images = np.sum(ud_center_masked_images, axis=2)
                ud_center_masked_images = torch.from_numpy(ud_center_masked_images).cuda()
                depth_mask = ud_center_masked_images > torch.mean(ud_center_masked_images)  # avoid close objects
                ud_center_masked_images = ud_center_masked_images * depth_mask.float()

                undistort_bad_indices = (
                    ud_center_masked_images > 1.0
                ).nonzero()  # bad_uv_idx, viewpoint_cam, depth_map, gt_image, new_ray_step=3

                # median_depth = torch.median(depth)
                diff_sorted, _ = torch.sort(depth.reshape(-1))
                N = diff_sorted.shape[0]
                median_depth = int(0.7 * N)
                median_depth = diff_sorted[median_depth]

                depth = torch.where(depth > median_depth, depth, median_depth)
                if optim_args.shuffle_ems == 0:
                    total_N_new_points = gaussians.add_gaussians(
                        undistort_bad_indices,
                        viewpoint_cam,
                        depth,
                        gt_image_x2_ud_torch.squeeze(0),
                        new_ray_step=optim_args.new_ray_step,
                        ray_end=optim_args.ray_end,
                        depth_max=depth_dict[viewpoint_cam.image_name],
                    )
                else:
                    total_N_new_points = gaussians.add_gaussians(
                        undistort_bad_indices,
                        viewpoint_cam,
                        depth,
                        gt_image_x2_ud_torch.squeeze(0),
                        new_ray_step=optim_args.new_ray_step,
                        ray_end=optim_args.ray_end,
                        depth_max=depth_dict[viewpoint_cam.image_name],
                        shuffle=True,
                    )

                scene.record_points(iteration, "depth" + str(torch.max(depth).item()))

                ud_center_masked_images_binary = ud_center_masked_images > 1.0
                ud_center_masked_images_binary = ud_center_masked_images_binary.float()
                gt_image = gt_image_x2_ud_torch * ud_center_masked_images_binary
                image = render_pkg["render"] * error_mask

                scene.record_points(iteration, "after add_points_by_uv" + viewpoint_cam.image_name)

                torchvision.utils.save_image(
                    ud_center_masked_images, os.path.join(path_dir, "masked_undisted_mask" + str(iteration) + ".png")
                )

                torchvision.utils.save_image(
                    gt_image, os.path.join(path_dir, "masked_ud_gt" + str(iteration) + ".png")
                )
                torchvision.utils.save_image(image, os.path.join(path_dir, "masked_render" + str(iteration) + ".png"))
                visibility_filter = torch.cat((visibility_filter, torch.zeros(total_N_new_points).cuda(0)), dim=0)
                visibility_filter = visibility_filter.bool()
                radii = torch.cat((radii, torch.zeros(total_N_new_points).cuda(0)), dim=0)
                viewspace_point_tensor = torch.cat(
                    (viewspace_point_tensor, torch.zeros(total_N_new_points, 3).cuda(0)), dim=0
                )

            if iteration < optim_args.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


if __name__ == "__main__":
    # Set up command line argument parser;
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)  # we put more parameters in optimization params, just for convenience.
    pp = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6029)
    parser.add_argument("--debug_from", type=int, default=-2)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument(
        "--save_iterations", nargs="+", type=int, default=[7_000, 10000, 12000, 15000, 20_000, 25_000, 30_000]
    )
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[20_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--densify", type=int, default=1, help="densify =1, we control points on N3d model_args")
    parser.add_argument("--duration", type=int, default=50, help="5 debug , 50 used")
    parser.add_argument("--basic_function", type=str, default="gaussian")
    parser.add_argument("--rgb_function", type=str, default="rgbv1")
    parser.add_argument("--rd_pipe", type=str, default="v2")
    parser.add_argument("--config_path", type=str, default="None")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)  # important !!!! seed 0,

    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # incase we provide config file not directly pass to the file     config will overwrite the argument... to change to the reverse?
    if os.path.exists(args.config_path) and args.config_path != "None":
        print("overload config from " + args.config_path)
        config = json.load(open(args.config_path))
        for k in config.keys():
            try:
                value = getattr(args, k)
                newvalue = config[k]
                setattr(args, k, newvalue)
            except:
                print("failed set config: " + k)
        print("finish load config from " + args.config_path)
    else:
        print("config file not exist or not provided")

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # refactor the code may affect results? unsure. keep the original structure
    args.iterations = 20000  # hard coded do not change

    train(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        densify=args.densify,
        duration=args.duration,
        basic_function=args.basic_function,
        rgb_function=args.rgb_function,
        rd_pipe=args.rd_pipe,
    )

    # All done
    print("\nTraining complete.")
