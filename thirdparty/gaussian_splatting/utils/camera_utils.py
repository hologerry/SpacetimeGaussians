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

import os

import numpy as np
import torch

from thirdparty.gaussian_splatting.scene.cameras import Camera_ss  # ass
from thirdparty.gaussian_splatting.scene.cameras import Camera
from thirdparty.gaussian_splatting.utils.general_utils import pil_to_torch
from thirdparty.gaussian_splatting.utils.graphics_utils import fov2focal


WARNED = False


def load_cam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w / (resolution_scale * args.resolution)), round(
            orig_h / (resolution_scale * args.resolution)
        )
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print(
                        "[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1"
                    )
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = pil_to_torch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    camera_direct = cam_info.hp_directions
    camera_pose = cam_info.pose

    if camera_pose is not None:
        raise NotImplementedError("get_image_rays is not implemented")
        c2w = torch.FloatTensor(camera_pose)
        rays_o, rays_d = get_image_rays(camera_direct, c2w)  # get_image_rays          get_rays
    else:
        rays_o = None
        rays_d = None
    return Camera(
        colmap_id=cam_info.uid,
        R=cam_info.R,
        T=cam_info.T,
        FoVx=cam_info.FovX,
        FoVy=cam_info.FovY,
        image=gt_image,
        gt_alpha_mask=loaded_mask,
        image_name=cam_info.image_name,
        uid=id,
        data_device=args.data_device,
        near=cam_info.near,
        far=cam_info.far,
        timestamp=cam_info.timestamp,
        rayo=rays_o,
        rayd=rays_d,
    )


def load_cam_v2(args, id, cam_info, resolution_scale):

    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w / (resolution_scale * args.resolution)), round(
            orig_h / (resolution_scale * args.resolution)
        )
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print(
                        "[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1"
                    )
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = pil_to_torch(cam_info.image, resolution)
    resized_image_rgb_real = pil_to_torch(cam_info.real_image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    gt_image_real = resized_image_rgb_real[:3, ...]
    loaded_mask = None
    loaded_mask_real = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    if resized_image_rgb_real.shape[1] == 4:
        loaded_mask_real = resized_image_rgb_real[3:4, ...]

    camera_direct = cam_info.hp_directions
    camera_pose = cam_info.pose

    if camera_pose is not None:
        rays_o, rays_d = 1, camera_direct
    else:
        rays_o = None
        rays_d = None
    return Camera(
        colmap_id=cam_info.uid,
        R=cam_info.R,
        T=cam_info.T,
        FoVx=cam_info.FovX,
        FoVy=cam_info.FovY,
        image=gt_image,
        gt_alpha_mask=loaded_mask,
        image_name=cam_info.image_name,
        uid=id,
        data_device=args.data_device,
        near=cam_info.near,
        far=cam_info.far,
        time_idx=cam_info.time_idx,
        timestamp=cam_info.timestamp,
        rayo=rays_o,
        rayd=rays_d,
        cxr=cam_info.cxr,
        cyr=cam_info.cyr,
        is_fake_view=cam_info.is_fake_view,
        real_image=gt_image_real,
        gt_alpha_mask_real=loaded_mask_real,
    )


def load_cam_v2_timing(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w / (resolution_scale * args.resolution)), round(
            orig_h / (resolution_scale * args.resolution)
        )
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print(
                        "[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1"
                    )
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = pil_to_torch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    camera_direct = cam_info.hp_directions
    camera_pose = cam_info.pose

    if camera_pose is not None:
        rays_o, rays_d = 1, camera_direct
    else:
        rays_o = None
        rays_d = None

    return Camera(
        colmap_id=cam_info.uid,
        R=cam_info.R,
        T=cam_info.T,
        FoVx=cam_info.FovX,
        FoVy=cam_info.FovY,
        image=gt_image,
        gt_alpha_mask=loaded_mask,
        image_name=cam_info.image_name,
        uid=id,
        data_device=args.data_device,
        near=cam_info.near,
        far=cam_info.far,
        timestamp=cam_info.timestamp,
        rayo=rays_o,
        rayd=rays_d,
        cxr=cam_info.cxr,
        cyr=cam_info.cyr,
    )


def load_cam_v2_ss(args, id, cam_info, resolution_scale, no_gt=False):
    if not no_gt:
        orig_w, orig_h = cam_info.image.size
    else:
        orig_w, orig_h = cam_info.width, cam_info.height

    assert args.resolution == 1
    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w / (resolution_scale * args.resolution)), round(
            orig_h / (resolution_scale * args.resolution)
        )
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print(
                        "[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1"
                    )
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resolution = (int(orig_w / 2), int(orig_h / 2))

    camera_direct = cam_info.hp_directions
    camera_pose = cam_info.pose
    loaded_mask = None

    # load gt image
    if no_gt == False:
        resized_image_rgb = pil_to_torch(cam_info.image, resolution)  # hard coded half resolution

        gt_image = resized_image_rgb[:3, ...]

        #
        if "01_Welder" in args.source_path:
            if "camera_0009" in cam_info.image_name:
                gt_image = gt_image / 1.15

        if "12_Cave" in args.source_path:
            if "camera_0009" in cam_info.image_name:
                gt_image = gt_image * 1.15
                gt_image = gt_image.clamp(0.0, 1.0)

        if "04_Truck" in args.source_path:
            if "camera_0008" in cam_info.image_name:
                gt_image = gt_image / 1.2

        if resized_image_rgb.shape[1] == 4:
            loaded_mask = resized_image_rgb[3:4, ...]

        if camera_pose is not None:
            rays_o, rays_d = 1, camera_direct
        else:
            rays_o = None
            rays_d = None
        return Camera_ss(
            colmap_id=cam_info.uid,
            R=cam_info.R,
            T=cam_info.T,
            FoVx=cam_info.FovX,
            FoVy=cam_info.FovY,
            image=gt_image,
            gt_alpha_mask=loaded_mask,
            image_name=cam_info.image_name,
            uid=id,
            data_device=args.data_device,
            near=cam_info.near,
            far=cam_info.far,
            timestamp=cam_info.timestamp,
            rayo=rays_o,
            rayd=rays_d,
            cxr=cam_info.cxr,
            cyr=cam_info.cyr,
        )
    else:

        if camera_pose is not None:
            rays_o, rays_d = 1, camera_direct
        else:
            rays_o = None
            rays_d = None
        return Camera_ss(
            colmap_id=cam_info.uid,
            R=cam_info.R,
            T=cam_info.T,
            FoVx=cam_info.FovX,
            FoVy=cam_info.FovY,
            image=resolution,
            gt_alpha_mask=loaded_mask,
            image_name=cam_info.image_name,
            uid=id,
            data_device=args.data_device,
            near=cam_info.near,
            far=cam_info.far,
            timestamp=cam_info.timestamp,
            rayo=rays_o,
            rayd=rays_d,
            cxr=cam_info.cxr,
            cyr=cam_info.cyr,
        )


def load_cam_no_gt(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.width, cam_info.height

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w / (resolution_scale * args.resolution)), round(
            orig_h / (resolution_scale * args.resolution)
        )
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print(
                        "[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1"
                    )
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = None  # pil_to_torch(cam_info.image, resolution)

    gt_image = None  # resized_image_rgb[:3, ...]
    loaded_mask = None

    loaded_mask = None

    camera_direct = cam_info.hp_directions
    camera_pose = cam_info.pose

    rays_o, rays_d = 1, camera_direct

    return Camera(
        colmap_id=cam_info.uid,
        R=cam_info.R,
        T=cam_info.T,
        FoVx=cam_info.FovX,
        FoVy=cam_info.FovY,
        image=resolution,
        gt_alpha_mask=loaded_mask,
        image_name=cam_info.image_name,
        uid=id,
        data_device=args.data_device,
        near=cam_info.near,
        far=cam_info.far,
        timestamp=cam_info.timestamp,
        rayo=rays_o,
        rayd=rays_d,
    )


def camera_list_from_cam_infos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(load_cam(args, id, c, resolution_scale))


def camera_list_from_cam_infos_v2(cam_infos, resolution_scale, args, ss=False):
    camera_list = []

    if not ss:
        for id, c in enumerate(cam_infos):
            camera_list.append(load_cam_v2(args, id, c, resolution_scale))
    else:
        for id, c in enumerate(cam_infos):
            camera_list.append(load_cam_v2_ss(args, id, c, resolution_scale))
            print("id", id)

    return camera_list


def camera_list_from_cam_infos_v2_no_gt(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(load_cam_no_gt(args, id, c, resolution_scale))

    return camera_list


def camera_to_json(id, camera: Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        "id": id,
        "img_name": camera.image_name,
        "width": camera.width,
        "height": camera.height,
        "position": pos.tolist(),
        "rotation": serializable_array_2d,
        "fy": fov2focal(camera.FovY, camera.height),
        "fx": fov2focal(camera.FovX, camera.width),
    }
    return camera_entry
