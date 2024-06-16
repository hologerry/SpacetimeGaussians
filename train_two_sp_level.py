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

import os
import random

from argparse import Namespace
from random import randint

import lovely_tensors as lt
import numpy as np
import torch

from torchvision.utils import save_image
from tqdm import tqdm

from gaussian_splatting.arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_splatting.gaussian.gaussian_model import GaussianModel
from gaussian_splatting.helper3dg import get_render_parts
from gaussian_splatting.scene import Scene
from gaussian_splatting.utils.graphics_utils import get_world_2_view2
from gaussian_splatting.utils.image_utils import psnr
from gaussian_splatting.utils.loss_utils import l1_loss, ssim
from helper_gaussian import get_model
from helper_parser import get_parser, write_args_to_file
from helper_pipe import get_render_pipe
from helper_train import (
    control_gaussians,
    prepare_output_and_logger,
    reload_helper,
    trb_exp_linear_function,
    trb_function,
)
from image_video_io import images_to_video


def train(
    args: Namespace,
    model_args: ModelParams,
    optim_args: OptimizationParams,
    pipe_args: PipelineParams,
    testing_iterations: list,
    saving_iterations: list,
    checkpoint_iterations: list,
    debug_from: int,
):
    write_args_to_file(args, model_args, optim_args, pipe_args, "training")

    tb_writer = prepare_output_and_logger(model_args)
    first_iter = 0
    render_func, GRsetting, GRzer = get_render_pipe(pipe_args.rd_pipe)

    print(f"Model: {model_args.model}")
    Gaussian = get_model(model_args.model)

    # trbf means Temporal Radial Basis Function in the paper
    # the trbf_center µ^τ_i is the temporal center, trbf_scale s^τ_i is temporal scaling factor
    gaussians: GaussianModel = Gaussian(model_args.sh_degree, model_args.rgb_function)
    gaussians.trbf_scale_init = -1 * optim_args.trbf_scale_init
    gaussians.preprocess_points = optim_args.preprocess_points
    gaussians.add_sph_points_scale = optim_args.add_sph_points_scale
    gaussians.ray_start = optim_args.ray_start

    if "opacity_exp_linear" in pipe_args.rd_pipe:
        print("Using opacity_exp_linear TRBF for opacity")
        trbf_base_function = trb_exp_linear_function
    else:
        trbf_base_function = trb_function

    scene = Scene(
        model_args,
        gaussians,
        loader=model_args.loader,
    )

    current_xyz = gaussians._xyz
    # os.makedirs("vis_cam", exist_ok=True)
    # np.save(os.path.join("vis_cam", "input_xyz.npy"), current_xyz.detach().cpu().numpy())
    # z wrong... # ???
    max_x, max_y, max_z = torch.amax(current_xyz[:, 0]), torch.amax(current_xyz[:, 1]), torch.amax(current_xyz[:, 2])
    min_x, min_y, min_z = torch.amin(current_xyz[:, 0]), torch.amin(current_xyz[:, 1]), torch.amin(current_xyz[:, 2])

    if os.path.exists(optim_args.prev_path):
        print("load from " + optim_args.prev_path)
        reload_helper(gaussians, optim_args, max_x, max_y, max_z, min_x, min_y, min_z)

    max_bounds = [max_x, max_y, max_z]
    min_bounds = [min_x, min_y, min_z]

    gaussians.training_setup(optim_args)

    num_channel = 3 if model_args.grey_image else 1

    bg_color = [1, 1, 1] if model_args.white_background else [0 for i in range(num_channel)]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    # viewpoint_stack = None
    ema_loss_for_log = 0.0
    # if freeze != 1:
    first_iter = 0
    progress_bar = tqdm(range(first_iter, optim_args.iterations), desc="Training progress")
    first_iter += 1

    flag = 0

    train_camera_list = scene.get_train_cameras().copy()
    train_cam_dict = {}
    unique_timestamps = sorted(list(set([cam.timestamp for cam in train_camera_list])))

    for i, timestamp in enumerate(unique_timestamps):
        train_cam_dict[i] = [cam for cam in train_camera_list if cam.timestamp == timestamp]

    if gaussians.ts is None:
        H, W = train_camera_list[0].image_height, train_camera_list[0].image_width
        gaussians.ts = torch.ones(1, 1, H, W).cuda()

    scene.record_points(0, "start training")

    # loss_dict = {}
    # ssim_dict = {}
    # depth_dict = {}
    # depth_dict = {}
    # valid_depth_dict = {}
    # ems_start_from_iterations = optim_args.ems_start  # guided sampling start from iteration

    with torch.no_grad():
        time_index = 0
        viewpoint_set = train_cam_dict[time_index]
        for viewpoint_cam in viewpoint_set:
            render_pkg = render_func(
                viewpoint_cam,
                gaussians,
                pipe_args,
                background,
                override_color=None,
                basic_function=trbf_base_function,
                GRsetting=GRsetting,
                GRzer=GRzer,
            )

            w2c = get_world_2_view2(viewpoint_cam.R, viewpoint_cam.T)
            c2w = np.linalg.inv(w2c)
            c2w[:3, 1:3] *= -1
            # os.makedirs("vis_cam", exist_ok=True)
            # np.save(f"vis_cam/cam_{viewpoint_cam.image_name}_pose.npy", c2w)

            # _, depthH, depthW = render_pkg["depth"].shape
            # border_H = int(depthH / 2)
            # border_W = int(depthW / 2)

            # mid_h = int(viewpoint_cam.image_height / 2)
            # mid_w = int(viewpoint_cam.image_width / 2)

            depth = render_pkg["depth"]
            print(f"Cam {viewpoint_cam.image_name} initial depth: {depth}")
            select_mask = depth != 15.0
            select_mask_sum = torch.sum(select_mask)

            initial_image = render_pkg["render"]
            gt_image = viewpoint_cam.original_image.float().cuda()
            # valid_depth_dict[viewpoint_cam.image_name] = torch.median(depth[select_mask]).item()
            # depth_dict[viewpoint_cam.image_name] = torch.amax(depth[select_mask]).item()
            save_image(initial_image, os.path.join(scene.model_path, f"initial_render_{viewpoint_cam.image_name}.png"))
            save_image(gt_image, os.path.join(scene.model_path, f"initial_gt_{viewpoint_cam.image_name}.png"))

            assert select_mask_sum > 0, f"No valid depth for {viewpoint_cam.image_name}"

    # lpips_criteria = lpips.LPIPS(net="alex", verbose=False).cuda()

    print(f"Level 0 clone: {optim_args.clone}, split {optim_args.split}, prune {optim_args.prune}")

    for iteration in range(first_iter, optim_args.iterations + 1):

        cur_level = 0 if iteration < optim_args.level_1_start_iter else 1

        if iteration == optim_args.level_1_start_iter:
            gaussians.create_another_level(
                spatial_lr_scale=scene.cameras_extent,
                new_pts_per_time=model_args.level_1_init_num_pts_per_time,
                new_pts_init_op=model_args.level_1_init_pts_op,
                new_pts_init_color=model_args.level_1_init_pts_color,
                new_pts_init_xyz=model_args.level_1_init_pts_xyz,
                new_pts_init_xyz_offset=model_args.level_1_init_pts_xyz_offset,
                new_pts_init_scale=model_args.level_1_init_pts_scale,
                new_pts_init_min_opacity=model_args.level_1_init_pts_min_opacity,
                new_pts_init_delta_rot_radius_scale=model_args.level_1_init_pts_delta_rot_radius_scale,
                new_pts_init_delta_rot_angle_vel_rand=model_args.level_1_init_pts_delta_rot_angle_vel_rand,
                new_pts_fix_trbfs=model_args.level_1_init_pts_fix_trbfs,
                new_pts_init_delta_sin_a=model_args.level_1_delta_sin_a,
                new_pts_init_delta_sin_omega=model_args.level_1_delta_sin_omega,
                new_pts_init_delta_sin_phi=model_args.level_1_delta_sin_phi,
                new_pts_init_per_parent=model_args.level_1_init_num_pts_per_parent,
                new_pts_init_delta_rig_sur_radius_scale=model_args.level_1_init_pts_delta_rig_sur_radius_scale,
                new_pts_init_delta_x_max=model_args.level_1_init_delta_x_max,
                new_pts_init_delta_y_max=model_args.level_1_init_delta_y_max,
                new_pts_init_delta_z_max=model_args.level_1_init_delta_z_max,
                start_time=model_args.start_time,
                duration=model_args.duration,
                time_step=model_args.time_step,
            )
            gaussians.level_1_training_setup(optim_args)

        iter_start.record()
        if cur_level == 0:
            gaussians.update_learning_rate(iteration)
        elif cur_level == 1:
            gaussians.update_level_1_learning_rate(iteration - optim_args.level_1_start_iter)

        if (iteration - 1) == debug_from:
            pipe_args.debug = True

        if cur_level == 0:
            gaussians.zero_gradient_cache()
        elif cur_level == 1:
            gaussians.zero_level_1_gradient_cache()

        time_index = randint(0, (len(unique_timestamps) - 1))
        viewpoint_set = train_cam_dict[time_index]
        cam_index = random.sample(viewpoint_set, optim_args.batch)

        # loss = 0.0
        for i in range(optim_args.batch):
            viewpoint_cam = cam_index[i]
            render_pkg = render_func(
                viewpoint_cam,
                gaussians,
                pipe_args,
                background,
                override_color=None,
                basic_function=trbf_base_function,
                GRsetting=GRsetting,
                GRzer=GRzer,
                level=cur_level,
                act_level_1=optim_args.act_level_1,
                transp_level_0=optim_args.transparent_level_0,
                rotdel_type=model_args.level_1_delta_rot_type,
            )
            image, viewspace_point_tensor, visibility_filter, radii = get_render_parts(render_pkg)
            # depth = render_pkg["depth"]
            # print(f"iter: {iteration}, batch: {i}")
            # print(f"image: {image.shape} {image.min()} {image.max()}")
            # print(
            #     f"viewspace_point_tensor: {viewspace_point_tensor.shape} {viewspace_point_tensor.min()} {viewspace_point_tensor.max()}"
            # )
            # print(f"visibility_filter: {visibility_filter.shape} {visibility_filter.min()} {visibility_filter.max()}")
            # print(f"radii: {radii.shape} {radii.min()} {radii.max()}")
            # print(f"depth: {depth.shape} {depth.min()} {depth.max()}")

            # print(f"viewpoint_cam {viewpoint_cam.image_name} {viewpoint_cam.is_fake_view}")

            gt_image = viewpoint_cam.original_image.float().cuda()
            # gt_image_real = viewpoint_cam.original_image_real.float().cuda()

            # if iteration > model_args.level_1_start_iter:
            #     level_0_render_pkg = render_func(
            #         viewpoint_cam,
            #         gaussians,
            #         pipe_args,
            #         background,
            #         override_color=None,
            #         basic_function=trbf_base_function,
            #         GRsetting=GRsetting,
            #         GRzer=GRzer,
            #         level=0.0
            #     )
            #     lvl0_image, lvl0_viewspace_point_tensor, lvl0_visibility_filter, lvl0_radii = get_render_parts(level_0_render_pkg)

            # if optim_args.gt_mask:
            #     # for training with undistorted immersive image, masking black pixels in undistorted image.
            #     mask = torch.sum(gt_image, dim=0) == 0
            #     mask = mask.float()
            #     image = image * (1 - mask) + gt_image * (mask)

            if iteration == optim_args.level_1_start_iter:
                save_image(
                    image, os.path.join(scene.model_path, f"level_1_initial_render_{viewpoint_cam.image_name}.png")
                )
                save_image(
                    gt_image, os.path.join(scene.model_path, f"level_1_initial_gt_{viewpoint_cam.image_name}.png")
                )

            view_name = viewpoint_cam.image_name

            l1_loss_value = l1_loss(image, gt_image)
            ssim_loss_value = 1.0 - ssim(image, gt_image)
            weight_loss = (1.0 - optim_args.lambda_dssim) * l1_loss_value + optim_args.lambda_dssim * ssim_loss_value
            loss = weight_loss

            # if iteration > model_args.level_1_start_iter:
            #     lvl0_l1_loss_value = l1_loss(lvl0_image, gt_image)
            #     lvl0_ssim_loss_value = 1.0 - ssim(lvl0_image, gt_image)
            #     lvl0_weight_loss = (1.0 - optim_args.lambda_dssim) * lvl0_l1_loss_value + optim_args.lambda_dssim * lvl0_ssim_loss_value
            #     loss += lvl0_weight_loss

            if optim_args.lambda_velocity > 0:
                velocities3D = render_pkg["velocities3D"]
                loss_velocity_x = torch.abs(velocities3D[:, 0]).mean()
                loss_velocity_y = torch.abs(velocities3D[:, 1]).mean()
                loss_velocity_z = torch.abs(velocities3D[:, 2]).mean()
                # less regularization on y
                loss_velocity = loss_velocity_x + 0.5 * loss_velocity_y + loss_velocity_z
                velocity_loss = optim_args.lambda_velocity * loss_velocity
                loss += velocity_loss

            if optim_args.lambda_opacity_vel > 0:
                opacity_vel = render_pkg["opacity_vel"]
                opacity_vel = torch.abs(opacity_vel)
                loss_opacity_velocity = opacity_vel.mean()
                opacity_velocity_loss = optim_args.lambda_opacity_vel * loss_opacity_velocity
                loss += opacity_velocity_loss

            if optim_args.lambda_level_1_motion > 0 and cur_level == 1:
                level_1_motion = gaussians.get_level_1_motion
                level_1_motion = torch.abs(level_1_motion)
                loss_level_1_motion = level_1_motion.mean()
                level_1_motion_loss = optim_args.lambda_level_1_motion * loss_level_1_motion
                loss += level_1_motion_loss

            if optim_args.lambda_level_1_delta_xyz > 0 and cur_level == 1:
                level_1_delta_xyz = render_pkg["level_1_delta_means3D"]
                level_1_delta_xyz = torch.abs(level_1_delta_xyz)
                loss_level_1_delta_xyz = level_1_delta_xyz.mean()
                level_1_delta_xyz_loss = optim_args.lambda_level_1_delta_xyz * loss_level_1_delta_xyz
                loss += level_1_delta_xyz_loss

            if optim_args.lambda_level_1_delta_xyz_smooth > 0 and cur_level == 1:
                level_1_delta_xyz = render_pkg["level_1_delta_means3D"]
                level_1_delta_xyz_prev = gaussians.get_level_1_delta_xyz(viewpoint_cam.time_idx - 1)
                level_1_delta_smooth = torch.abs(level_1_delta_xyz - level_1_delta_xyz_prev)
                loss_level_1_delta_xyz_smooth = level_1_delta_smooth.mean()
                level_1_delta_xyz_smooth_loss = optim_args.lambda_level_1_delta_xyz_smooth * loss_level_1_delta_xyz_smooth
                loss += level_1_delta_xyz_smooth_loss

            tb_writer.add_scalar(f"train_loss/l1_loss_{view_name}", l1_loss_value.item(), iteration)
            tb_writer.add_scalar(f"train_loss/ssim_loss_{view_name}", ssim_loss_value.item(), iteration)
            tb_writer.add_scalar(f"train_loss/w_loss_{view_name}", weight_loss.item(), iteration)
            tb_writer.add_scalar(f"train_loss/total_loss_{view_name}", loss.item(), iteration)
            # if iteration > model_args.level_1_start_iter:
            #     tb_writer.add_scalar(f"train_loss/l1_loss_lvl0_{view_name}", lvl0_l1_loss_value.item(), iteration)
            #     tb_writer.add_scalar(f"train_loss/ssim_loss_lvl0_{view_name}", lvl0_ssim_loss_value.item(), iteration)
            #     tb_writer.add_scalar(f"train_loss/w_loss_lvl0_{view_name}", lvl0_weight_loss.item(), iteration)

            if optim_args.lambda_velocity > 0:
                tb_writer.add_scalar(f"train_loss/vel_loss_{view_name}", velocity_loss.item(), iteration)
            if optim_args.lambda_opacity_vel > 0:
                tb_writer.add_scalar(
                    f"train_loss/opacity_vel_loss_{view_name}", opacity_velocity_loss.item(), iteration
                )
            if optim_args.lambda_level_1_motion > 0 and cur_level == 1:
                tb_writer.add_scalar(
                    f"train_loss/level_1_motion_loss_{view_name}", level_1_motion_loss.item(), iteration
                )
            if optim_args.lambda_level_1_delta_xyz > 0 and cur_level == 1:
                tb_writer.add_scalar(
                    f"train_loss/level_1_delta_xyz_loss_{view_name}", level_1_delta_xyz_loss.item(), iteration
                )

            if optim_args.lambda_level_1_delta_xyz_smooth > 0 and cur_level == 1:
                tb_writer.add_scalar(
                    f"train_loss/level_1_delta_xyz_smooth_loss_{view_name}", level_1_delta_xyz_smooth_loss.item(), iteration
                )

            loss.backward()
            if cur_level == 0:
                gaussians.cache_gradient()
                gaussians.optimizer.zero_grad(set_to_none=True)
            elif cur_level == 1:
                gaussians.cache_level_1_gradient()
                gaussians.level_1_optimizer.zero_grad(set_to_none=True)

        iter_end.record()
        if cur_level == 0:
            gaussians.set_batch_gradient(optim_args.batch)
        elif cur_level == 1:
            gaussians.set_level_1_batch_gradient(optim_args.batch)

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if iteration % 10 == 0:
                post_fix = {"Level": f"{cur_level}", "Loss": f"{ema_loss_for_log:.7f}"}
                if cur_level == 0:
                    num_points = gaussians.get_xyz.shape[0]
                    post_fix["Points"] = num_points
                elif cur_level == 1:
                    # since in two_sp_level_couple model, xyz is fake
                    num_level_0_points = gaussians.get_xyz.shape[0]
                    num_level_1_points = gaussians.get_level_1_features.shape[0]
                    post_fix["Points0"] = num_level_0_points
                    post_fix["Points1"] = num_level_1_points

                progress_bar.set_postfix(post_fix)
                progress_bar.update(10)

            if iteration == optim_args.iterations:
                progress_bar.close()

            if iteration in saving_iterations:
                print(f"[ITER {iteration}] Saving Gaussians")
                scene.record_points(iteration, "saving", two_level=True)
                scene.save(iteration)

            # Log and save
            training_report(
                model_args,
                optim_args,
                tb_writer,
                iteration,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render_func,
                (pipe_args, background),
                trbf_base_function,
                GRsetting,
                GRzer,
                pipe_args.rd_pipe,
                # test_all_train_views=True,
                cur_level=cur_level,
            )
            if cur_level == 0:
                cur_clone = optim_args.clone
                cur_split = optim_args.split
                cur_split_prune = optim_args.split_prune
                cur_prune = optim_args.prune
                # cur_zero_grad_level = None
            elif cur_level == 1:
                cur_clone = optim_args.level_1_clone
                cur_split = optim_args.level_1_split
                cur_split_prune = optim_args.level_1_split_prune
                cur_prune = optim_args.level_1_prune
                # cur_zero_grad_level = None

            if cur_level == 1 and iteration == optim_args.level_1_start_iter:
                print(f"Level 1 clone: {cur_clone}, split {cur_split}, prune {cur_prune}")

            # Densification and pruning here
            flag = control_gaussians(
                optim_args,
                gaussians,
                model_args.densify,
                iteration,
                scene,
                visibility_filter,
                radii,
                viewspace_point_tensor,
                flag,
                train_camera_with_distance=None,
                max_bounds=max_bounds,
                min_bounds=min_bounds,
                white_background=model_args.white_background,
                # max_timestamp=model_args.max_timestamp,
                clone=cur_clone,
                split=cur_split,
                split_prune=cur_split_prune,
                prune=cur_prune,
                level=cur_level,
            )

        # gaussians.zero_gradient_by_level(cur_zero_grad_level)

        # Optimizer step
        if iteration < optim_args.iterations:
            if cur_level == 0:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
            elif cur_level == 1:
                gaussians.level_1_optimizer.step()
                gaussians.level_1_optimizer.zero_grad(set_to_none=True)

            # if iteration in checkpoint_iterations:
            #     print(f"\n[ITER {iteration}] Saving Checkpoint")
            #     torch.save((gaussians.capture(), iteration), scene.model_path + f"/ckp" + str(iteration) + ".pth")


def training_report(
    model_args,
    optim_args,
    tb_writer,
    iteration,
    elapsed,
    testing_iterations,
    scene,
    render_func,
    render_args,
    trbf_base_function,
    GRsetting,
    GRzer,
    rd_pipe,
    test_all_train_views=False,
    cur_level=0,
):
    if tb_writer:
        tb_writer.add_scalar("iter_time", elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        train_cams = scene.get_train_cameras()
        if not test_all_train_views:
            ids = [idx for idx in range(0, len(train_cams), 100)]
            train_cams = [train_cams[idx] for idx in ids]
        validation_configs = (
            {"name": "test", "cameras": scene.get_test_cameras()},
            {
                "name": "train",
                "cameras": train_cams,
            },
        )

        for config in validation_configs:
            l1_test = 0.0
            psnr_test = 0.0
            all_view_names = set()
            for idx, viewpoint in enumerate(config["cameras"]):
                rendered = render_func(
                    viewpoint,
                    scene.gaussians,
                    *render_args,
                    override_color=None,
                    basic_function=trbf_base_function,
                    GRsetting=GRsetting,
                    GRzer=GRzer,
                    level=cur_level,
                    act_level_1=optim_args.act_level_1,
                    transp_level_0=optim_args.transparent_level_0,
                    rotdel_type=model_args.level_1_delta_rot_type,
                )
                all_view_names.add(viewpoint.image_name)
                image = torch.clamp(rendered["render"], 0.0, 1.0)
                gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                # testing view it is no difference
                # training view real is the real gt, no real is the fake gt
                gt_image_real = torch.clamp(viewpoint.original_image_real.to("cuda"), 0.0, 1.0)
                # if hasattr(viewpoint, "original_image_real"):
                #     # since we using fake view images, we need to compare with real images
                #     gt_image = torch.clamp(viewpoint.original_image_real.to("cuda"), 0.0, 1.0)
                save_image(
                    image,
                    os.path.join(
                        scene.model_path,
                        "training_render",
                        f"render_{viewpoint.image_name}_{viewpoint.uid:03d}_{iteration:05d}.png",
                    ),
                )
                save_image(
                    gt_image,
                    os.path.join(
                        scene.model_path,
                        "training_render",
                        f"gt_{viewpoint.image_name}_{viewpoint.uid:03d}_{iteration:05d}.png",
                    ),
                )
                # save_image(
                #     gt_image_real,
                #     os.path.join(
                #         scene.model_path,
                #         "training_render",
                #         f"gt_real_{viewpoint.image_name}_{viewpoint.uid:03d}_{iteration:05d}.png",
                #     ),
                # )
                if tb_writer and (idx < 5):
                    tb_writer.add_images(
                        config["name"] + f"_view_{viewpoint.image_name}/render",
                        image[None],
                        global_step=iteration,
                    )
                    if iteration == testing_iterations[0]:
                        tb_writer.add_images(
                            config["name"] + f"_view_{viewpoint.image_name}/ground_truth",
                            gt_image[None],
                            global_step=iteration,
                        )
                        # tb_writer.add_images(
                        #     config["name"] + f"_view_{viewpoint.image_name}/ground_truth_real",
                        #     gt_image_real[None],
                        #     global_step=iteration,
                        # )
                l1_test += l1_loss(image, gt_image_real).mean().double()
                psnr_test += psnr(image, gt_image_real).mean().double()

            for view_name in list(all_view_names):
                images_to_video(
                    os.path.join(scene.model_path, "training_render"),
                    f"render_{view_name}",
                    f"{iteration:05d}.png",
                    os.path.join(scene.model_path, f"training_render_{view_name}_{iteration:05d}.mp4"),
                    fps=30,
                )
                images_to_video(
                    os.path.join(scene.model_path, "training_render"),
                    f"gt_{view_name}",
                    f"{iteration:05d}.png",
                    os.path.join(scene.model_path, f"training_gt_{view_name}_{iteration:05d}.mp4"),
                    fps=30,
                )
                # images_to_video(
                #     os.path.join(scene.model_path, "training_render"),
                #     f"gt_real_{view_name}",
                #     f"{iteration:05d}.png",
                #     os.path.join(scene.model_path, f"training_gt_real_{view_name}_{iteration:05d}.mp4"),
                #     fps=30,
                # )

            psnr_test /= len(config["cameras"])
            l1_test /= len(config["cameras"])
            print(f"[ITER {iteration}] Evaluating {config['name']}: L1 {l1_test} PSNR {psnr_test}")

            if tb_writer:
                tb_writer.add_scalar(config["name"] + "/loss_viewpoint - l1_loss", l1_test, iteration)
                tb_writer.add_scalar(config["name"] + "/loss_viewpoint - psnr", psnr_test, iteration)

        if tb_writer:
            if "opacity_linear" in rd_pipe:
                tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians._opacity, iteration)
            else:
                tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar("total_points", scene.gaussians.get_xyz.shape[0], iteration)

        torch.cuda.empty_cache()


if __name__ == "__main__":
    lt.monkey_patch()
    args, mp_extract, op_extract, pp_extract = get_parser()
    train(
        args,
        mp_extract,
        op_extract,
        pp_extract,
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.debug_from,
    )

    # All done
    print("Training complete.")
