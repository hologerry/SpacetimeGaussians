#
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

# =============================================

# This license is additionally subject to the following restrictions:

# Licensor grants non-exclusive rights to use the Software for research purposes
# to research users (both academic and industrial), free of charge, without right
# to sublicense. The Software may be used "non-commercially", i.e., for research
# and/or evaluation purposes only.

# Subject to the terms and conditions of this License, you are granted a
# non-exclusive, royalty-free, license to reproduce, prepare derivative works of,
# publicly display, publicly perform and distribute its Work and any resulting
# derivative works in any form.
#

import json
import os
import uuid

from argparse import Namespace

import cv2
import numpy as np
import torch

from script.pre_immersive_distorted import SCALE_DICT


try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print(f"Output folder: {args.model_path}")
    os.makedirs(args.model_path, exist_ok=True)
    os.makedirs(os.path.join(args.model_path, "training_render"), exist_ok=True)

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def freeze_weights(model, key_list):
    for k in key_list:
        grad_tensor = getattr(getattr(model, k), "grad")
        new_grad = torch.zeros_like(grad_tensor)
        setattr(getattr(model, k), "grad", new_grad)
    return


def freeze_weights_by_mask(model, key_list, mask):
    for k in key_list:
        grad_tensor = getattr(getattr(model, k), "grad")
        new_grad = mask.unsqueeze(1) * grad_tensor  # torch.zeros_like(grad_tensor)
        setattr(getattr(model, k), "grad", new_grad)
    return


def freeze_weights_by_mask_no_unsqueeze(model, key_list, mask):
    for k in key_list:
        grad_tensor = getattr(getattr(model, k), "grad")
        new_grad = mask * grad_tensor  # torch.zeros_like(grad_tensor)
        setattr(getattr(model, k), "grad", new_grad)
    return


def remove_min_max(gaussians, max_bounds, min_bounds):
    max_x, max_y, max_z = max_bounds
    min_x, min_y, min_z = min_bounds
    xyz = gaussians._xyz
    mask0 = xyz[:, 0] > max_x.item()
    mask1 = xyz[:, 1] > max_y.item()
    mask2 = xyz[:, 2] > max_z.item()

    mask3 = xyz[:, 0] < min_x.item()
    mask4 = xyz[:, 1] < min_y.item()
    mask5 = xyz[:, 2] < min_z.item()
    mask = logical_or_list([mask0, mask1, mask2, mask3, mask4, mask5])
    gaussians.prune_points(mask)
    torch.cuda.empty_cache()


def control_gaussians(
    opt,
    gaussians,
    densify,
    iteration,
    scene,
    visibility_filter,
    radii,
    viewspace_point_tensor,
    flag,
    train_camera_with_distance=None,
    max_bounds=None,
    min_bounds=None,
    white_background=False,
    # max_timestamp=1.0,
    clone=True,
    split=True,
    split_prune=True,
    prune=True,
    level=0,
):
    if level == 0 and iteration < opt.densify_until_iter:
        gaussians.max_radii2D[visibility_filter] = torch.max(
            gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
        )
        gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

    if level == 1 and iteration < opt.level_1_densify_until_iter:
        if opt.transparent_level_0:
            # for transparent level 0, the viewspace_point_tensor and visibility_filter are in level 1
            level_1_radii = radii
            level_1_visibility_filter = visibility_filter
        else:
            n_level_0_points = gaussians.get_xyz.shape[0]
            level_1_radii = radii[n_level_0_points:]
            level_1_visibility_filter = visibility_filter[n_level_0_points:]

        gaussians.level_1_max_radii2D[level_1_visibility_filter] = torch.max(
            gaussians.level_1_max_radii2D[level_1_visibility_filter], level_1_radii[level_1_visibility_filter]
        )
        gaussians.add_densification_stats_level_1(viewspace_point_tensor, visibility_filter)


    if densify == 0:  # raw gaussian splatting

        if level == 0 and iteration < opt.densify_until_iter:

            if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                gaussians.densify_and_prune(
                    opt.densify_grad_threshold,
                    0.005,
                    scene.cameras_extent,
                    size_threshold,
                    # max_timestamp=max_timestamp,
                    clone=clone,
                    split=split,
                    split_prune=split_prune,
                    prune=prune,
                )

            if iteration % opt.opacity_reset_interval == 0 or (
                white_background and iteration == opt.densify_from_iter
            ):
                gaussians.reset_opacity()

            ## bbox pruning not working, cause optimizer state to None, directly use the range to initialize points
            # gaussians.prune_points_bbox(scene.bbox_model)

        if level == 1 and iteration < opt.level_1_densify_until_iter:

            if iteration > opt.level_1_densify_from_iter and iteration % opt.densification_interval == 0:
                size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                gaussians.densify_and_prune_level_1(
                    opt.densify_grad_threshold,
                    0.005,
                    scene.cameras_extent,
                    size_threshold,
                    # max_timestamp=max_timestamp,
                    clone=clone,
                    split=split,
                    split_prune=split_prune,
                    prune=prune,
                )

            if iteration % opt.opacity_reset_interval == 0 or (
                white_background and iteration == opt.level_1_densify_from_iter
            ):
                gaussians.reset_opacity_level_1()

        if (
            level == 0
            and opt.post_prune
            and iteration > opt.post_prune_from_iter
            and iteration % opt.post_prune_interval == 0
            and iteration < opt.post_prune_until_iter
        ):
            size_threshold = 20 if iteration > opt.opacity_reset_interval else None
            gaussians.post_prune(
                0.005,
                scene.cameras_extent,
                size_threshold,
            )

        if (level == 1
            and opt.level_1_post_prune
            and iteration > opt.level_1_post_prune_from_iter
            and iteration % opt.level_1_post_prune_interval == 0
            and iteration < opt.level_1_post_prune_until_iter
        ):
            size_threshold = 20 if iteration > opt.opacity_reset_interval else None
            gaussians.post_prune_level_1(
                0.005,
                scene.cameras_extent,
                size_threshold,
                opt.level_1_post_prune_min_color,
                opt.level_1_post_prune_max_color,
            )

        return flag

    elif densify == 1:  # n3d
        if iteration < opt.densify_until_iter:
            if iteration == 8001:
                omega_mask = gaussians.zero_omega_by_motion()  # 1 we keep omega, 0 we freeze omega
                gaussians.omega_mask = omega_mask
                scene.record_points(iteration, "separate omega" + str(torch.sum(omega_mask).item()))
            elif iteration > 8001:
                freeze_weights_by_mask_no_unsqueeze(gaussians, ["_omega"], gaussians.omega_mask)
                rotation_mask = torch.logical_not(gaussians.omega_mask)
                freeze_weights_by_mask_no_unsqueeze(gaussians, ["_rotation"], rotation_mask)
            if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                if flag < opt.densify_cnt:
                    scene.record_points(iteration, "before densify")
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_prune_clone(
                        opt.densify_grad_threshold, opt.opacity_threshold, scene.cameras_extent, size_threshold
                    )
                    flag += 1
                    scene.record_points(iteration, "after densify")
                else:
                    if iteration < 7000:  # default 7000.
                        prune_mask = (gaussians.get_opacity < opt.opacity_threshold).squeeze()
                        gaussians.prune_points(prune_mask)
                        torch.cuda.empty_cache()
                        scene.record_points(iteration, "additionally prune_mask")
            if iteration % opt.opacity_reset_interval == 0:
                gaussians.reset_opacity()
        else:
            freeze_weights_by_mask_no_unsqueeze(gaussians, ["_omega"], gaussians.omega_mask)
            rotation_mask = torch.logical_not(gaussians.omega_mask)
            freeze_weights_by_mask_no_unsqueeze(
                gaussians, ["_rotation"], rotation_mask
            )  # uncomment freeze weight ... for fast training speed.
            if iteration % 1000 == 500:
                z_mask = gaussians._xyz[:, 2] < 4.5  #
                gaussians.prune_points(z_mask)
                torch.cuda.empty_cache()
            if iteration == 10000:
                remove_min_max(gaussians, max_bounds, min_bounds)
        return flag

    elif densify == 2:  # n3d
        if iteration < opt.densify_until_iter:
            if iteration == 8001:  # 8001
                omega_mask = gaussians.zero_omega_by_motion()  #
                gaussians.omega_mask = omega_mask
                scene.record_points(iteration, "separate omega" + str(torch.sum(omega_mask).item()))
            elif iteration > 8001:  # 8001
                freeze_weights_by_mask_no_unsqueeze(gaussians, ["_omega"], gaussians.omega_mask)
                rotation_mask = torch.logical_not(gaussians.omega_mask)
                freeze_weights_by_mask_no_unsqueeze(gaussians, ["_rotation"], rotation_mask)
            if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                if flag < opt.densify_cnt:
                    scene.record_points(iteration, "before densify")
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_prune_clone(
                        opt.densify_grad_threshold, opt.opacity_threshold, scene.cameras_extent, size_threshold
                    )
                    flag += 1
                    scene.record_points(iteration, "after densify")
                else:
                    prune_mask = (gaussians.get_opacity < opt.opacity_threshold).squeeze()
                    gaussians.prune_points(prune_mask)
                    torch.cuda.empty_cache()
                    scene.record_points(iteration, "additionally prune_mask")
            if iteration % 3000 == 0:
                gaussians.reset_opacity()
        else:
            if iteration % 1000 == 500:
                z_mask = gaussians._xyz[:, 2] < 4.5  # for stability
                gaussians.prune_points(z_mask)
                torch.cuda.empty_cache()
        return flag

    elif densify == 3:  # techni
        if iteration < opt.densify_until_iter:
            gaussians.max_radii2D[visibility_filter] = torch.max(
                gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
            )
            gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

            if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                if flag < opt.densify_cnt:
                    scene.record_points(iteration, "before densify")
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_prune_clone(
                        opt.densify_grad_threshold, opt.opacity_threshold, scene.cameras_extent, size_threshold
                    )
                    flag += 1
                    scene.record_points(iteration, "after densify")
                else:
                    if iteration < 7000:  # default 7000.
                        prune_mask = (gaussians.get_opacity < opt.opacity_threshold).squeeze()
                        gaussians.prune_points(prune_mask)
                        torch.cuda.empty_cache()
                        scene.record_points(iteration, "additionally prune_mask")
            if iteration % opt.opacity_reset_interval == 0:
                gaussians.reset_opacity()
        else:
            if iteration == 10000:
                remove_min_max(gaussians, max_bounds, min_bounds)
        return flag


def logical_or_list(tensor_list):
    mask = None
    for idx, ele in enumerate(tensor_list):
        if idx == 0:
            mask = ele

        else:
            mask = torch.logical_or(mask, ele)
    return mask


def record_points_helper(model_path, numpoints, iteration, string):
    txt_path = os.path.join(model_path, "N_points_log.txt")

    with open(txt_path, "a") as file:
        file.write("iteration at " + str(iteration) + "\n")
        file.write(string + " points number " + str(numpoints) + "\n")


def reload_helper(gaussians, opt, max_x, max_y, max_z, min_x, min_y, min_z):
    given_path = opt.prev_path
    if opt.load_all == 0:
        gaussians.load_ply_and_min_max(given_path, max_x, max_y, max_z, min_x, min_y, min_z)
    elif opt.load_all == 1:
        gaussians.load_ply_and_min_max_all(given_path, max_x, max_y, max_z, min_x, min_y, min_z)
    elif opt.load_all == 2:
        gaussians.load_ply(given_path)
    elif opt.load_all == 3:
        gaussians.load_ply_and_min_max_Y(given_path, max_x, max_y, max_z, min_x, min_y, min_z)

    gaussians.max_radii2D = torch.zeros((gaussians.get_xyz.shape[0]), device="cuda")
    return


def get_fish_eye_mapper(folder, camera_name):
    parent_folder = os.path.dirname(folder)
    distortion_flow_path = os.path.join(parent_folder, camera_name + ".npy")
    distortion_flow = np.load(distortion_flow_path)
    distortion_flow = torch.from_numpy(distortion_flow).unsqueeze(0).float().cuda()
    return distortion_flow


def undistort_image(image_name, dataset_path, data):

    video = os.path.dirname(dataset_path)  # upper folder

    with open(os.path.join(video + "/models.json"), "r") as f:
        meta = json.load(f)

    for idx, camera in enumerate(meta):
        folder = camera["name"]  # camera_0001
        view = camera
        intrinsics = np.array(
            [
                [view["focal_length"], 0.0, view["principal_point"][0]],
                [0.0, view["focal_length"], view["principal_point"][1]],
                [0.0, 0.0, 1.0],
            ]
        )
        dis_cef = np.zeros((4))

        dis_cef[:2] = np.array(view["radial_distortion"])[:2]
        if folder != image_name:
            continue
        print("done one camera")
        map1, map2 = None, None
        sequence_name = os.path.basename(video)
        focal_scale = SCALE_DICT[sequence_name]

        h, w = data.shape[:2]

        image_size = (w, h)
        knew = np.zeros((3, 3), dtype=np.float32)
        knew[0, 0] = focal_scale * intrinsics[0, 0]
        knew[1, 1] = focal_scale * intrinsics[1, 1]
        knew[0, 2] = view["principal_point"][0]  # cx fixed half of the width
        knew[1, 2] = view["principal_point"][1]  #
        knew[2, 2] = 1.0

        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            intrinsics, dis_cef, R=None, P=knew, size=(w, h), m1type=cv2.CV_32FC1
        )

        undistorted_image = cv2.remap(data, map1, map2, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
        undistorted_image = undistorted_image.clip(0, 255.0)
        return undistorted_image


def trb_function(x):
    # Temporal Radial Basis Function
    return torch.exp(-1 * x.pow(2))


def trb_exp_linear_function(x):
    # Temporal Radial Basis Function
    return torch.exp(-1 * x)
