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
import sys
import time
import warnings

from os import makedirs

import lovely_tensors as lt
import numpy as np
import scipy
import torch
import torch.nn.functional as F
import torchvision

from imageio import mimsave
from skimage.metrics import structural_similarity as sk_ssim
from torchvision.utils import save_image
from tqdm import tqdm

from helper_train import get_model, get_render_pipe, trb_function, trb_exp_linear_function
from image_video_io import images_to_video, mp4_to_gif
from thirdparty.gaussian_splatting.arguments import ModelParams, PipelineParams
from thirdparty.gaussian_splatting.helper3dg import get_test_parser
from thirdparty.gaussian_splatting.lpipsPyTorch import lpips as lpips_func
from thirdparty.gaussian_splatting.scene import Scene
from thirdparty.gaussian_splatting.utils.image_utils import psnr as psnr_func
from thirdparty.gaussian_splatting.utils.loss_utils import ssim as ssim_func


warnings.filterwarnings("ignore")


# modified from https://github.com/graphdeco-inria/gaussian-splatting/blob/main/render.py
# and https://github.com/graphdeco-inria/gaussian-splatting/blob/main/metrics.py
def render_set(
    model_path,
    name,
    iteration,
    views,
    gaussians,
    pipe_args,
    background,
    trbf_base_function,
    rd_pipe,
    grey_image=False,
):

    if "velocity" in rd_pipe:
        post_str = "_vel"
    elif "vis" in rd_pipe:
        post_str = "_vis"
    else:
        post_str = ""
    print(f"post_str {post_str}")

    render_path = os.path.join(model_path, name, f"ours_{iteration}{post_str}", "renders")
    gts_path = os.path.join(model_path, name, f"ours_{iteration}{post_str}", "gt")
    quantities_out_path = os.path.join(model_path, name, f"ours_{iteration}{post_str}", "quantities")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(quantities_out_path, exist_ok=True)

    if gaussians.rgb_decoder is not None:
        gaussians.rgb_decoder.cuda()
        gaussians.rgb_decoder.eval()

    stats_dict = {}

    scales = gaussians.get_scaling

    scales_max = torch.amax(scales).item()
    scales_mean = torch.mean(scales).item()

    stats_dict["scales_max"] = scales_max
    stats_dict["scales_mean"] = scales_mean

    if "opacity_linear" not in rd_pipe:
        op = gaussians.get_opacity
        op_max = torch.amax(op).item()
        op_mean = torch.mean(op).item()
        stats_dict["op_max"] = op_max
        stats_dict["op_mean"] = op_mean

    stats_path = os.path.join(model_path, "stat_" + str(iteration) + ".json")
    with open(stats_path, "w") as fp:
        json.dump(stats_dict, fp, indent=True)

    test_rd_pipe = rd_pipe.replace("train", "test")
    test_rd_pipe = test_rd_pipe.replace("_full_ss", "_full_ss_fused")
    test_rd_pipe += "_vis"
    render, GRsetting, GRzer = get_render_pipe(test_rd_pipe)

    img_channel = 1 if grey_image else 3

    full_metrics_dict = {}

    all_view_names = []

    for idx, view in enumerate(tqdm(views, desc="Rendering and metric progress")):
        rendering_pkg = render(
            view,
            gaussians,
            pipe_args,
            background,
            scaling_modifier=1.0,
            basic_function=trbf_base_function,
            GRsetting=GRsetting,
            GRzer=GRzer,
        )
        rendering = rendering_pkg["render"]
        cur_view_name = view.image_name
        cur_view_time_idx = view.time_idx
        if cur_view_name not in full_metrics_dict:
            all_view_names.append(cur_view_name)
            full_metrics_dict[cur_view_name] = {}
            full_metrics_dict[cur_view_name]["l1"] = []
            full_metrics_dict[cur_view_name]["ssim"] = []
            full_metrics_dict[cur_view_name]["psnr"] = []
            full_metrics_dict[cur_view_name]["lpips"] = []
            full_metrics_dict[cur_view_name]["lpips_vgg"] = []
            full_metrics_dict[cur_view_name]["ssim_v2"] = []

        rendering = torch.clamp(rendering, 0, 1.0)
        gt = view.original_image[0:img_channel, :, :].cuda().float()

        ssim = ssim_func(rendering.unsqueeze(0), gt.unsqueeze(0)).item()
        l1 = F.l1_loss(rendering.unsqueeze(0), gt.unsqueeze(0)).item()
        psnr = psnr_func(rendering.unsqueeze(0), gt.unsqueeze(0)).mean().item()
        lpips = lpips_func(rendering.unsqueeze(0), gt.unsqueeze(0), net_type="alex").mean().item()
        lpips_vgg = lpips_func(rendering.unsqueeze(0), gt.unsqueeze(0), net_type="vgg").mean().item()

        render_numpy = rendering.permute(1, 2, 0).detach().cpu().numpy()
        gt_numpy = gt.permute(1, 2, 0).detach().cpu().numpy()
        ssim_v2 = sk_ssim(
            render_numpy, gt_numpy, channel_axis=-1, multichannel=True, data_range=gt_numpy.max() - gt_numpy.min()
        )

        full_metrics_dict[cur_view_name]["l1"].append(l1)
        full_metrics_dict[cur_view_name]["ssim"].append(ssim)
        full_metrics_dict[cur_view_name]["psnr"].append(psnr)
        full_metrics_dict[cur_view_name]["lpips"].append(lpips)
        full_metrics_dict[cur_view_name]["lpips_vgg"].append(lpips_vgg)
        full_metrics_dict[cur_view_name]["ssim_v2"].append(ssim_v2)

        os.makedirs(os.path.join(render_path, cur_view_name), exist_ok=True)
        os.makedirs(os.path.join(gts_path, cur_view_name), exist_ok=True)

        rendering_image_save_path = os.path.join(render_path, cur_view_name, f"{cur_view_time_idx:05d}.png")
        gt_image_save_path = os.path.join(gts_path, cur_view_name, f"{cur_view_time_idx:05d}.png")
        save_image(rendering, rendering_image_save_path)
        save_image(gt, gt_image_save_path)

        if cur_view_name == "train02":

            means3D_no_t = rendering_pkg["means3D_no_t"]
            means3D_no_t = means3D_no_t.detach().cpu().numpy()
            means3D_no_t_path = os.path.join(quantities_out_path, f"means3D_no_t_{cur_view_time_idx:05d}.npy")
            np.save(means3D_no_t_path, means3D_no_t)


            means3D = rendering_pkg["means3D"]
            means3D = means3D.detach().cpu().numpy()
            means3D_path = os.path.join(quantities_out_path, f"means3D_{cur_view_time_idx:05d}.npy")
            np.save(means3D_path, means3D)

            if "means3D_timed" in rendering_pkg:
                means3D_timed = rendering_pkg["means3D_timed"]
                means3D_timed = means3D_timed.detach().cpu().numpy()
                means3D_timed_path = os.path.join(quantities_out_path, f"means3D_timed_{cur_view_time_idx:05d}.npy")
                np.save(means3D_timed_path, means3D_timed)

            if "means3D_zeroed" in rendering_pkg:
                means3D_zeroed = rendering_pkg["means3D_zeroed"]
                means3D_zeroed = means3D_zeroed.detach().cpu().numpy()
                means3D_zeroed_path = os.path.join(quantities_out_path, f"means3D_zeroed_{cur_view_time_idx:05d}.npy")
                np.save(means3D_zeroed_path, means3D_zeroed)

            trbf_center = rendering_pkg["trbf_center"]
            trbf_center = trbf_center.detach().cpu().numpy()
            trbf_center_path = os.path.join(quantities_out_path, f"trbf_center_{cur_view_time_idx:05d}.npy")
            np.save(trbf_center_path, trbf_center)

            velocities3D = rendering_pkg["velocities3D"]
            velocities3D = velocities3D.detach().cpu().numpy()
            velocities3D_path = os.path.join(quantities_out_path, f"velocities3D_{cur_view_time_idx:05d}.npy")
            np.save(velocities3D_path, velocities3D)

            if "velocities3D_timed" in rendering_pkg:
                velocities3D_timed = rendering_pkg["velocities3D_timed"]
                velocities3D_timed = velocities3D_timed.detach().cpu().numpy()
                velocities3D_timed_path = os.path.join(quantities_out_path, f"velocities3D_timed_{cur_view_time_idx:05d}.npy")
                np.save(velocities3D_timed_path, velocities3D_timed)

            if "velocities3D_zeroed" in rendering_pkg:
                velocities3D_zeroed = rendering_pkg["velocities3D_zeroed"]
                velocities3D_zeroed = velocities3D_zeroed.detach().cpu().numpy()
                velocities3D_zeroed_path = os.path.join(quantities_out_path, f"velocities3D_zeroed_{cur_view_time_idx:05d}.npy")
                np.save(velocities3D_zeroed_path, velocities3D_zeroed)

            opacity = rendering_pkg["opacity"]
            opacity = opacity.detach().cpu().numpy()
            opacity_path = os.path.join(quantities_out_path, f"opacity_{cur_view_time_idx:05d}.npy")
            np.save(opacity_path, opacity)

            if "opacity_timed" in rendering_pkg:
                opacity_timed = rendering_pkg["opacity_timed"]
                opacity_timed = opacity_timed.detach().cpu().numpy()
                opacity_timed_path = os.path.join(quantities_out_path, f"opacity_timed_{cur_view_time_idx:05d}.npy")
                np.save(opacity_timed_path, opacity_timed)

            if "opacity_zeroed" in rendering_pkg:
                opacity_zeroed = rendering_pkg["opacity_zeroed"]
                opacity_zeroed = opacity_zeroed.detach().cpu().numpy()
                opacity_zeroed_path = os.path.join(quantities_out_path, f"opacity_zeroed_{cur_view_time_idx:05d}.npy")
                np.save(opacity_zeroed_path, opacity_zeroed)

            visibility_filter = rendering_pkg["visibility_filter"]
            visibility_filter = visibility_filter.detach().cpu().numpy()
            visibility_filter_path = os.path.join(
                quantities_out_path, f"visibility_filter_{cur_view_time_idx:05d}.npy"
            )
            np.save(visibility_filter_path, visibility_filter)

            radii = rendering_pkg["radii"]
            radii = radii.detach().cpu().numpy()
            radii_path = os.path.join(quantities_out_path, f"radii_{cur_view_time_idx:05d}.npy")
            np.save(radii_path, radii)

            motion = rendering_pkg["motion"]
            motion = motion.detach().cpu().numpy()
            motion_path = os.path.join(quantities_out_path, f"motion_{cur_view_time_idx:05d}.npy")
            np.save(motion_path, motion)

            rotations = rendering_pkg["rotations"]
            rotations = rotations.detach().cpu().numpy()
            rotations_path = os.path.join(quantities_out_path, f"rotations_{cur_view_time_idx:05d}.npy")
            np.save(rotations_path, rotations)

            colors_precomp = rendering_pkg["colors_precomp"]
            colors_precomp = colors_precomp.detach().cpu().numpy()
            colors_precomp_path = os.path.join(quantities_out_path, f"colors_precomp_{cur_view_time_idx:05d}.npy")
            np.save(colors_precomp_path, colors_precomp)

            scales = rendering_pkg["scales"]
            scales = scales.detach().cpu().numpy()
            scales_path = os.path.join(quantities_out_path, f"scales_{cur_view_time_idx:05d}.npy")
            np.save(scales_path, scales)

            R_path = os.path.join(quantities_out_path, f"R_{cur_view_time_idx:05d}.npy")
            T_path = os.path.join(quantities_out_path, f"T_{cur_view_time_idx:05d}.npy")
            np.save(R_path, view.R)
            np.save(T_path, view.T)

            fov_path = os.path.join(quantities_out_path, f"fov_{cur_view_time_idx:05d}.npy")
            fov = np.array([view.FoVx, view.FoVy])
            np.save(fov_path, fov)

    print(f"all_view_names: {all_view_names}")
    for view_name in all_view_names:
        render_frame_path = os.path.join(render_path, view_name)
        gt_frame_path = os.path.join(gts_path, view_name)
        render_out_mp4_path = os.path.join(model_path, name + f"_render_{view_name}_{iteration}{post_str}.mp4")
        gt_out_mp4_path = os.path.join(model_path, name + f"_gt_{view_name}_{iteration}{post_str}.mp4")
        images_to_video(render_frame_path, "", ".png", render_out_mp4_path, fps=30)
        images_to_video(gt_frame_path, "", ".png", gt_out_mp4_path, fps=30)
        mp4_to_gif(render_out_mp4_path, render_out_mp4_path.replace(".mp4", ".gif"))
        mp4_to_gif(gt_out_mp4_path, gt_out_mp4_path.replace(".mp4", ".gif"))

    mean_metrics_per_view_dict = {}
    for view_name in all_view_names:
        mean_metrics_per_view_dict[view_name] = {}
        mean_metrics_per_view_dict[view_name]["l1"] = float(np.mean(full_metrics_dict[view_name]["l1"]))
        mean_metrics_per_view_dict[view_name]["ssim"] = float(np.mean(full_metrics_dict[view_name]["ssim"]))
        mean_metrics_per_view_dict[view_name]["psnr"] = float(np.mean(full_metrics_dict[view_name]["psnr"]))
        mean_metrics_per_view_dict[view_name]["lpips"] = float(np.mean(full_metrics_dict[view_name]["lpips"]))
        mean_metrics_per_view_dict[view_name]["lpips_vgg"] = float(np.mean(full_metrics_dict[view_name]["lpips_vgg"]))
        mean_metrics_per_view_dict[view_name]["ssim_v2"] = float(np.mean(full_metrics_dict[view_name]["ssim_v2"]))


    if "train00" in all_view_names:
        train_view_names = ["train00", "train01", "train03", "train04"]
        train_mean_metrics = {}
        for metric in ["l1", "ssim", "psnr", "lpips", "lpips_vgg", "ssim_v2"]:
            train_mean_metrics[metric] = float(
                np.mean([np.mean(full_metrics_dict[view_name][metric]) for view_name in train_view_names])
            )

    test_view_names = ["train02"]
    test_mean_metrics = {}
    for metric in ["l1", "ssim", "psnr", "lpips", "lpips_vgg", "ssim_v2"]:
        test_mean_metrics[metric] = float(
            np.mean([np.mean(full_metrics_dict[view_name][metric]) for view_name in test_view_names])
        )

    # for idx, view in enumerate(tqdm(views, desc="release gt images cuda memory for timing")):
    #     view.original_image = None  # .detach()
    #     torch.cuda.empty_cache()

    ## start timing

    # for idx, view in enumerate(tqdm(views, desc="timing ")):
    #     for _ in range(20):
    #         render_pack = render(
    #             view,
    #             gaussians,
    #             pipe_args,
    #             background,
    #             scaling_modifier=1.0,
    #             basic_function=trbf_base_function,
    #             GRsetting=GRsetting,
    #             GRzer=GRzer,
    #         )  # ["time"] # C x H x W
    #         duration = render_pack["duration"]
    #         if idx > 10:  # warm up
    #             times.append(duration)

    # print("mean time for rendering", np.mean(np.array(times)))

    if "train00" in all_view_names:
        with open(model_path + "/" + str(iteration) + "_train_views.json", "w") as fp:
            print("Saving train views results")
            json.dump(train_mean_metrics, fp, indent=True)

    with open(model_path + "/" + str(iteration) + "_test_views.json", "w") as fp:
        print("Saving test views results")
        json.dump(test_mean_metrics, fp, indent=True)

    if "train00" in all_view_names:
        with open(model_path + "/" + str(iteration) + "_per_view.json", "w") as fp:
            print("Saving per view results")
            json.dump(mean_metrics_per_view_dict, fp, indent=True)


# render free view
def render_set_no_gt(
    model_path,
    name,
    iteration,
    views,
    gaussians,
    pipe_args,
    background,
    trbf_base_function,
    rd_pipe,
    grey_image=False,
):
    render, GRsetting, GRzer = get_render_pipe(rd_pipe)
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")

    makedirs(render_path, exist_ok=True)
    if gaussians.rgb_decoder is not None:
        gaussians.rgb_decoder.cuda()
        gaussians.rgb_decoder.eval()

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # C x H x W
        rendering = render(
            view,
            gaussians,
            pipe_args,
            background,
            scaling_modifier=1.0,
            basic_function=trbf_base_function,
            GRsetting=GRsetting,
            GRzer=GRzer,
        )
        rendered_image = rendering["render"]

        save_image(rendered_image, os.path.join(render_path, "{0:05d}".format(idx) + ".png"))


@torch.no_grad()
def run_test(
    model_args: ModelParams,
    iteration: int,
    pipe_args: PipelineParams,
    skip_train: bool,
    skip_test: bool,
    multi_view: bool,
    duration: int,
    rgb_function="rgbv1",
    rd_pipe="v2",
    loader="colmap",
    start_time=0,
    time_step=1,
):

    print("use model {}".format(model_args.model))
    GaussianModel = get_model(model_args.model)  # default, g_model, we are testing

    print(f"iteration {iteration}")
    gaussians = GaussianModel(model_args.sh_degree, rgb_function)

    scene = Scene(
        model_args,
        gaussians,
        load_iteration=iteration,
        shuffle=False,
        multi_view=multi_view,
        loader=loader,
        start_time=start_time,
        duration=duration,
        time_step=time_step,
        grey_image=model_args.grey_image,
        test_all_views=True,
    )
    if "opacity_exp_linear" in rd_pipe:
        print("Using opacity_exp_linear TRBF for opacity")
        trbf_base_function = trb_exp_linear_function
    else:
        trbf_base_function = trb_function

    num_channels = 9
    bg_color = [0 for _ in range(num_channels)]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    if gaussians.ts is None:
        camera_list = scene.get_test_cameras()
        H, W = camera_list[0].image_height, camera_list[0].image_width
        gaussians.ts = torch.ones(1, 1, H, W).cuda()

    if not skip_test and not multi_view:
        print("rendering test set")
        render_set(
            model_args.model_path,
            "test",
            scene.loaded_iter,
            scene.get_test_cameras(),
            gaussians,
            pipe_args,
            background,
            trbf_base_function,
            rd_pipe,
            grey_image=model_args.grey_image,
        )
    # if multi_view:
    #     print("rendering multi-view set no gt")
    #     render_set_no_gt(
    #         model_args.model_path,
    #         "mv",
    #         scene.loaded_iter,
    #         scene.get_test_cameras(),
    #         gaussians,
    #         pipe_args,
    #         background,
    #         trbf_base_function,
    #         rd_pipe,
    #         grey_image=model_args.grey_image,
    #     )


if __name__ == "__main__":
    lt.monkey_patch()

    args, model_args, pipe_args, multi_view = get_test_parser()
    run_test(
        model_args,
        args.test_iteration,
        pipe_args,
        args.skip_train,
        args.skip_test,
        multi_view,
        rgb_function=args.rgb_function,
        rd_pipe=args.rd_pipe,
        loader=args.val_loader,
        start_time=args.start_time,
        duration=args.duration,
        time_step=args.time_step,
    )
