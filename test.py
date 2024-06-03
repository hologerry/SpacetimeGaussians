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

import copy
import json
import os
import warnings

from os import makedirs

import lovely_tensors as lt
import numpy as np
import torch
import torch.nn.functional as F

from skimage.metrics import structural_similarity as sk_ssim
from torchvision.utils import save_image
from tqdm import tqdm

from gaussian_splatting.arguments import ModelParams, PipelineParams
from gaussian_splatting.lpipsPyTorch import lpips as lpips_func
from gaussian_splatting.scene import Scene
from gaussian_splatting.utils.image_utils import psnr as psnr_func
from gaussian_splatting.utils.loss_utils import ssim as ssim_func
from helper_gaussian import get_model
from helper_parser import get_test_parser
from helper_pipe import get_render_pipe
from helper_train import trb_exp_linear_function, trb_function
from image_video_io import images_to_video, mp4_to_gif


warnings.filterwarnings("ignore")


def save_quantities(rendering_pkg, cur_view_time_idx, out_path):
    for k, v in rendering_pkg.items():
        if isinstance(v, torch.Tensor):
            quantity = v.detach().cpu().numpy()
        elif isinstance(v, np.ndarray):
            quantity = v
        save_path = os.path.join(out_path, f"{k}_{cur_view_time_idx:05d}.npy")
        np.save(save_path, quantity)


@torch.no_grad()
def run_test(args, model_args: ModelParams, pipe_args: PipelineParams, iteration: int):

    model_path = model_args.model_path
    name = "test"

    print(f"Model: {model_args.model}")
    GaussianModel = get_model(model_args.model)

    print(f"Iteration: {iteration}")
    gaussians = GaussianModel(model_args.sh_degree, args.rgb_function)

    scene = Scene(
        model_args,
        gaussians,
        load_iteration=iteration,
        shuffle=False,
        multi_view=False,
        loader=args.val_loader,
        test_all_views=True,
    )

    trbf_base_function = trb_function

    if "opacity_exp_linear" in args.rd_pipe:
        print("Using opacity_exp_linear TRBF for opacity")
        trbf_base_function = trb_exp_linear_function

    if gaussians.ts is None:
        camera_list = scene.get_test_cameras()
        H, W = camera_list[0].image_height, camera_list[0].image_width
        gaussians.ts = torch.ones(1, 1, H, W).cuda()

    if "velocity" in args.rd_pipe:
        post_str = "_vel"
    elif "vis" in args.rd_pipe:
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

    test_rd_pipe = args.rd_pipe.replace("train", "test")
    test_rd_pipe += "_vis"
    render, GRsetting, GRzer = get_render_pipe(test_rd_pipe)

    img_channel = 1 if model_args.grey_image else 3

    bg_color = [0 for _ in range(img_channel)]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    metric_names = ["l1", "ssim", "psnr", "lpips", "lpips_vgg", "ssim_v2"]
    full_metrics_dict = {}

    all_view_names = []

    views = scene.get_test_cameras()
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
            for l_name in metric_names:
                full_metrics_dict[cur_view_name][l_name] = []

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
            render_numpy,
            gt_numpy,
            channel_axis=-1,
            multichannel=True,
            data_range=gt_numpy.max() - gt_numpy.min(),
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
            save_quantities(rendering_pkg, cur_view_time_idx, quantities_out_path)

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
        for l_name in metric_names:
            mean_metrics_per_view_dict[view_name][l_name] = float(np.mean(full_metrics_dict[view_name][l_name]))

    if "train00" in all_view_names:
        train_view_names = ["train00", "train01", "train03", "train04"]
        train_mean_metrics = {}
        for metric in metric_names:
            train_mean_metrics[metric] = float(
                np.mean([np.mean(full_metrics_dict[view_name][metric]) for view_name in train_view_names])
            )

    test_view_names = ["train02"]
    test_mean_metrics = {}
    for metric in metric_names:
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


@torch.no_grad()
def run_future(args, model_args: ModelParams, pipe_args: PipelineParams, iteration: int):

    model_path = model_args.model_path
    name = "future"

    print(f"Model: {model_args.model}")
    GaussianModel = get_model(model_args.model)

    print(f"Iteration: {iteration}")
    gaussians = GaussianModel(model_args.sh_degree, args.rgb_function)

    scene = Scene(
        model_args,
        gaussians,
        load_iteration=iteration,
        shuffle=False,
        multi_view=False,
        loader=args.val_loader,
        test_all_views=True,
    )

    trbf_base_function = trb_function

    if "opacity_exp_linear" in args.rd_pipe:
        print("Using opacity_exp_linear TRBF for opacity")
        trbf_base_function = trb_exp_linear_function

    if gaussians.ts is None:
        camera_list = scene.get_test_cameras()
        H, W = camera_list[0].image_height, camera_list[0].image_width
        gaussians.ts = torch.ones(1, 1, H, W).cuda()

    if "velocity" in args.rd_pipe:
        post_str = "_vel"
    elif "vis" in args.rd_pipe:
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

    test_rd_pipe = args.rd_pipe.replace("train", "test")
    test_rd_pipe += "_vis"
    render, GRsetting, GRzer = get_render_pipe(test_rd_pipe)

    img_channel = 1 if model_args.grey_image else 3

    bg_color = [0 for _ in range(img_channel)]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    metric_names = ["l1", "ssim", "psnr", "lpips", "lpips_vgg", "ssim_v2"]
    full_metrics_dict = {}

    all_view_names = []

    views = scene.get_test_cameras()

    future_view_start_idx = 0
    for idx, view in enumerate(tqdm(views, desc="Rendering training progress")):
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
            for l_name in metric_names:
                full_metrics_dict[cur_view_name][l_name] = []

        rendering = torch.clamp(rendering, 0, 1.0)
        gt = view.original_image[0:img_channel, :, :].cuda().float()

        os.makedirs(os.path.join(render_path, cur_view_name), exist_ok=True)
        os.makedirs(os.path.join(gts_path, cur_view_name), exist_ok=True)

        rendering_image_save_path = os.path.join(render_path, cur_view_name, f"{cur_view_time_idx:05d}.png")
        gt_image_save_path = os.path.join(gts_path, cur_view_name, f"{cur_view_time_idx:05d}.png")
        save_image(rendering, rendering_image_save_path)
        save_image(gt, gt_image_save_path)

        if cur_view_name == "train02":
            save_quantities(rendering_pkg, cur_view_time_idx, quantities_out_path)

        future_view_start_idx = max(future_view_start_idx, cur_view_time_idx)

    print(
        f"Test on future views future_view_start_idx: {future_view_start_idx}, max_timestamp: {model_args.max_timestamp}"
    )
    future_views = copy.deepcopy(views)
    for f_view in future_views:
        f_view.time_idx += future_view_start_idx
        f_view.timestamp += model_args.max_timestamp

    for idx, f_view in enumerate(tqdm(future_views, desc="Rendering future progress")):
        rendering_pkg = render(
            f_view,
            gaussians,
            pipe_args,
            background,
            scaling_modifier=1.0,
            basic_function=trbf_base_function,
            GRsetting=GRsetting,
            GRzer=GRzer,
        )
        rendering = rendering_pkg["render"]
        cur_view_name = f_view.image_name
        cur_view_time_idx = f_view.time_idx
        if cur_view_name not in full_metrics_dict:
            all_view_names.append(cur_view_name)
            full_metrics_dict[cur_view_name] = {}
            for l_name in metric_names:
                full_metrics_dict[cur_view_name][l_name] = []

        rendering = torch.clamp(rendering, 0, 1.0)

        os.makedirs(os.path.join(render_path, cur_view_name), exist_ok=True)

        rendering_image_save_path = os.path.join(render_path, cur_view_name, f"{cur_view_time_idx:05d}.png")
        save_image(rendering, rendering_image_save_path)

        if cur_view_name == "train02":
            save_quantities(rendering_pkg, cur_view_time_idx, quantities_out_path)

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


if __name__ == "__main__":
    lt.monkey_patch()

    args, model_args, pipe_args = get_test_parser()
    if args.test:
        run_test(args, model_args, pipe_args, args.test_iteration)
    if args.future:
        run_future(args, model_args, pipe_args, args.test_iteration)
