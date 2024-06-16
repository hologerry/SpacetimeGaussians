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
import torch.nn.functional as F

from torchvision.utils import save_image
from tqdm import tqdm

from gaussian_splatting.arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_splatting.gaussian.gm_simple_color_scale_rotation_act_two_sp_level_couple_transp_zerodel import (
    GaussianModel,
)
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


def simulate(
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

    sim_rd_pipe = pipe_args.rd_pipe.replace("train", "sim")
    sim_rd_pipe += "_vis"

    render_func, GRsetting, GRzer = get_render_pipe(pipe_args.rd_pipe)
    sim_render_func, sim_GRsetting, sim_GRzer = get_render_pipe(sim_rd_pipe)

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
        load_iteration=iteration,
        shuffle=False,
        multi_view=False,
        loader=model_args.loader,
    )

    current_xyz = gaussians._xyz
    # os.makedirs("vis_cam", exist_ok=True)
    # np.save(os.path.join("vis_cam", "input_xyz.npy"), current_xyz.detach().cpu().numpy())
    # z wrong... # ???
    # max_x, max_y, max_z = torch.amax(current_xyz[:, 0]), torch.amax(current_xyz[:, 1]), torch.amax(current_xyz[:, 2])
    # min_x, min_y, min_z = torch.amin(current_xyz[:, 0]), torch.amin(current_xyz[:, 1]), torch.amin(current_xyz[:, 2])

    # if os.path.exists(optim_args.prev_path):
    #     print("load from " + optim_args.prev_path)
    #     reload_helper(gaussians, optim_args, max_x, max_y, max_z, min_x, min_y, min_z)

    # max_bounds = [max_x, max_y, max_z]
    # min_bounds = [min_x, min_y, min_z]

    gaussians.simulation_setup(optim_args)

    num_channel = 3 if model_args.grey_image else 1

    bg_color = [1, 1, 1] if model_args.white_background else [0 for i in range(num_channel)]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    # viewpoint_stack = None
    ema_loss_for_log = 0.0
    # if freeze != 1:

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

    # flag_ems = 0
    # ems_cnt = 0
    # loss_dict = {}
    # ssim_dict = {}
    # depth_dict = {}
    # depth_dict = {}
    # valid_depth_dict = {}
    # ems_start_from_iterations = optim_args.ems_start  # guided sampling start from iteration

    with torch.no_grad():
        time_index = -1
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
                level=1,
                act_level_1=optim_args.act_level_1,
                transp_level_0=optim_args.transparent_level_0,
                rotdel_type=model_args.level_1_delta_rot_type,
            )

            depth = render_pkg["depth"]
            print(f"Cam {viewpoint_cam.image_name} initial depth: {depth}")
            select_mask = depth != 15.0
            select_mask_sum = torch.sum(select_mask)

            initial_image = render_pkg["render"]
            gt_image = viewpoint_cam.original_image.float().cuda()
            # valid_depth_dict[viewpoint_cam.image_name] = torch.median(depth[select_mask]).item()
            # depth_dict[viewpoint_cam.image_name] = torch.amax(depth[select_mask]).item()
            save_image(
                initial_image, os.path.join(scene.model_path, f"sim_init_render_{viewpoint_cam.image_name}.png")
            )
            save_image(gt_image, os.path.join(scene.model_path, f"sim_init_gt_{viewpoint_cam.image_name}.png"))

            assert select_mask_sum > 0, f"No valid depth for {viewpoint_cam.image_name}"

    init_means3D = render_pkg["means3D"]
    init_velocity = render_pkg["velocities3D"]
    point_nums = gaussians.sim_initialization_setup(optim_args, init_means3D, init_velocity)

    first_iter = 0
    progress_bar = tqdm(range(first_iter, optim_args.simulation_init_iters), desc="Simulation progress")
    first_iter += 1

    for iteration in range(first_iter, optim_args.simulation_init_iters + 1):
        iter_start.record()

        cur_point_idx = random.randint(0, point_nums - 1)
        sampled_xyz = gaussians.sim_sample_a_point(cur_point_idx)
        tilde_velocity_field = gaussians.sim_get_tilde_velocity_field(sampled_xyz)
        loss_val = F.l1_loss(tilde_velocity_field, init_velocity[cur_point_idx])

        loss = loss_val

        gaussians.sim_init_optimizer.zero_grad()
        loss.backward()
        gaussians.sim_init_optimizer.step()

        elapsed = iter_start.elapsed_time(iter_end)
        tb_writer.add_scalar(f"sim_init/iter_time", elapsed, iteration)
        tb_writer.add_scalar(f"sim_init_loss/l1_loss_val", loss_val.item(), iteration)
        ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
        if iteration % 10 == 0:
            post_fix = {"Loss": f"{ema_loss_for_log:.7f}"}
            progress_bar.set_postfix(post_fix)
            progress_bar.update(10)

        if iteration == optim_args.simulation_init_iters:
            progress_bar.close()

        if iteration in saving_iterations:
            print(f"[ITER {iteration}] Saving Gaussians")
            scene.record_points(iteration, "saving", two_level=True)
            scene.sim_save(iteration)

    # first_frame = 0
    # progress_bar = tqdm(range(first_frame, optim_args.simulation_frames), desc="Simulation progress")
    # first_frame += 1

    # for iteration in range(first_frame, optim_args.simulation_frames + 1):

    #     iter_start.record()

    #     if (iteration - 1) == debug_from:
    #         pipe_args.debug = True

    #     single_frame_delta_t = 1.0 / (len(unique_timestamps))

    #     gaussians.zero_si

    #     gaussians.simulation_advection_step(single_frame_delta_t)


def training_report(
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
    simulate(
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
