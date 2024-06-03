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

import glob
import json
import os
import sys

from pathlib import Path
from typing import NamedTuple

import cv2
import natsort
import numpy as np
import torch

from PIL import Image
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from tqdm import tqdm, trange

from gaussian_splatting.colmap_loader import (
    qvec_2_rot_mat,
    read_extrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_binary,
    read_intrinsics_text,
    read_points3D_binary,
    read_points3D_text,
)
from gaussian_splatting.utils.graphics_utils import (
    BasicPointCloud,
    focal2fov,
    fov2focal,
    get_projection_matrix,
    get_projection_matrix_cv,
    get_world_2_view2,
)
from gaussian_splatting.utils.sh_utils import sh2rgb

from .bbox_tool import BBoxTool


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    near: float
    far: float
    time_idx: int
    timestamp: float
    pose: np.array
    hp_directions: np.array
    cxr: float
    cyr: float
    is_fake_view: bool = False
    real_image: np.array = None


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    bbox_model: BBoxTool  # only used in hyfluid


def get_nerf_pp_norm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = get_world_2_view2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        # os.makedirs("cam_vis", exist_ok=True)
        # np.save(f"cam_vis/cam_{cam.image_name}_pose.npy", C2W)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def fetch_ply(path, grey_image=False):
    ply_data = PlyData.read(path)
    vertices = ply_data["vertex"]
    positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
    times = np.vstack([vertices["t"]]).T
    if grey_image:
        colors = np.vstack([vertices["grey"]]).T / 255.0
    else:
        colors = np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T / 255.0
    normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals, times=times)


def store_ply(path, xyzt, rgb, grey_image=False):
    # Define the dtype for the structured array
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("t", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
    ]
    if grey_image:
        dtype += [("grey", "u1")]
    else:
        dtype += [("red", "u1"), ("green", "u1"), ("blue", "u1")]

    xyz = xyzt[:, :3]
    normals = np.zeros_like(xyz)

    elements = np.empty(xyzt.shape[0], dtype=dtype)
    attributes = np.concatenate((xyzt, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, "vertex")
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def shift_image(image, offset_h, offset_w):
    shifted_image = np.zeros_like(image)

    # Perform the shift
    if offset_h > 0 and offset_w > 0:
        shifted_image[offset_h:, offset_w:, :] = image[:-offset_h, :-offset_w, :]
    elif offset_h > 0 and offset_w < 0:
        shifted_image[offset_h:, :offset_w, :] = image[:-offset_h, -offset_w:, :]
    elif offset_h < 0 and offset_w > 0:
        shifted_image[:offset_h, offset_w:, :] = image[-offset_h:, :-offset_w, :]
    elif offset_h < 0 and offset_w < 0:
        shifted_image[:offset_h, :offset_w, :] = image[-offset_h:, -offset_w:, :]
    elif offset_h > 0 and offset_w == 0:
        shifted_image[offset_h:, :, :] = image[:-offset_h, :, :]
    elif offset_h < 0 and offset_w == 0:
        shifted_image[:offset_h, :, :] = image[-offset_h:, :, :]
    elif offset_h == 0 and offset_w > 0:
        shifted_image[:, offset_w:, :] = image[:, :-offset_w, :]
    elif offset_h == 0 and offset_w < 0:
        shifted_image[:, :offset_w, :] = image[:, -offset_w:, :]

    return shifted_image


def read_cameras_from_transforms_hyfluid(
    path,
    transforms_file,
    white_background,
    extension=".png",
    start_time=50,
    duration=50,
    time_step=1,
    max_timestamp=1.0,
    grey_image=False,
    train_views="0134",
    train_views_fake=None,
    use_best_fake=False,
    img_offset=False,
):
    print(f"transforms_file: {transforms_file}, train_views: {train_views}, train_views_fake: {train_views_fake}")
    if img_offset:
        print("adding offset to image")
    cam_infos = []
    # print(f"start_time {start_time} duration {duration} time_step {time_step}")

    with open(os.path.join(path, transforms_file)) as json_file:
        contents = json.load(json_file)

    near = float(contents["near"])
    far = float(contents["far"])

    voxel_scale = np.array(contents["voxel_scale"])
    voxel_scale = np.broadcast_to(voxel_scale, [3])

    voxel_matrix = np.array(contents["voxel_matrix"])
    voxel_matrix = np.stack([voxel_matrix[:, 2], voxel_matrix[:, 1], voxel_matrix[:, 0], voxel_matrix[:, 3]], axis=1)
    voxel_matrix_inv = np.linalg.inv(voxel_matrix)
    bbox_model = BBoxTool(voxel_matrix_inv, voxel_scale)

    # voxel_R = -np.transpose(voxel_matrix[:3, :3])
    # voxel_R[:, 0] = -voxel_R[:, 0]
    # voxel_T = -voxel_matrix[:3, 3]

    frames = contents["frames"]
    camera_uid = 0
    for idx, frame in tqdm(enumerate(frames), desc="Reading views", total=len(frames), leave=False):  # camera idx

        # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = np.array(frame["transform_matrix"])
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        camera_hw = frame["camera_hw"]
        h, w = camera_hw
        fov_x = frame["camera_angle_x"]
        focal_length = fov2focal(fov_x, w)
        fov_y = focal2fov(focal_length, h)
        FovY = fov_y
        FovX = fov_x
        cam_name = frame["file_path"][-1:]  # train0x -> x used to determine with train_views
        # print(f"frame {frame['file_path']} focal_length {focal_length} FovX {FovX} FovY {FovY}")
        for time_idx in trange(start_time, start_time + duration, time_step, desc=f"cam0{cam_name}"):

            frame_name = os.path.join("colmap_frames", f"colmap_{time_idx}", frame["file_path"] + extension)
            # used to determine the loss type
            is_fake_view = False
            real_frame_name = frame_name

            if train_views_fake is not None and cam_name in train_views_fake:
                # print(f"FAKE VIEW: time_idx: {time_idx}, cam_name: {cam_name}, train_views_fake: {train_views_fake}")
                is_fake_view = True
                if use_best_fake:
                    frame_name = os.path.join(
                        f"zero123_finetune_15000_best_cam0{cam_name}_1920h_1080w", f"frame_{time_idx:06d}.png"
                    )
                else:
                    source_cam = train_views[:1]
                    frame_name = os.path.join(
                        f"zero123_finetune_15000_cam{source_cam}to{cam_name}_1920h_1080w",
                        f"frame_{time_idx:06d}.png",
                    )

            timestamp = (time_idx - start_time) / duration * max_timestamp

            image_path = os.path.join(path, frame_name)
            real_image_path = os.path.join(path, real_frame_name)
            # the image_name is used to index the camera so we all use the real name
            image_name = frame["file_path"].split("/")[-1]  # os.path.basename(image_path).split(".")[0]

            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            real_image = cv2.imread(real_image_path, cv2.IMREAD_COLOR)
            real_image = cv2.cvtColor(real_image, cv2.COLOR_BGR2RGB)

            if img_offset:
                if cam_name == "0":
                    image = shift_image(image, -12, 18)
                    real_image = shift_image(real_image, -12, 18)
                if cam_name == "1":
                    image = shift_image(image, 52, 18)
                    real_image = shift_image(real_image, 52, 18)
                if cam_name == "3":
                    image = shift_image(image, 11, -12)
                    real_image = shift_image(real_image, 11, -12)
                if cam_name == "4":
                    image = shift_image(image, 11, -18)
                    real_image = shift_image(real_image, 11, -18)

            image = Image.fromarray(image)
            real_image = Image.fromarray(real_image)

            if grey_image:
                image = image.convert("L")
                real_image = real_image.convert("L")

            pose = 1 if time_idx == start_time else None
            hp_directions = 1 if time_idx == start_time else None

            uid = camera_uid  # idx * duration//time_step + time_idx
            camera_uid += 1

            # print(f"frame_name {frame_name} timestamp {timestamp} camera uid {uid}")

            cam_infos.append(
                CameraInfo(
                    uid=uid,
                    R=R,
                    T=T,
                    FovY=FovY,
                    FovX=FovX,
                    image=image,
                    real_image=real_image,
                    image_path=image_path,
                    image_name=image_name,
                    width=image.size[0],
                    height=image.size[1],
                    time_idx=time_idx,
                    timestamp=timestamp,
                    near=near,
                    far=far,
                    pose=pose,
                    hp_directions=hp_directions,
                    cxr=0.0,
                    cyr=0.0,
                    is_fake_view=is_fake_view,
                )
            )

    return cam_infos, bbox_model


def read_nerf_synthetic_info_hyfluid(
    source_path,
    model_path,
    white_background,
    eval,
    extension=".png",
    start_time=50,
    duration=50,
    time_step=1,
    max_timestamp=1.0,
    grey_image=False,
    train_views="0134",
    train_views_fake=None,
    use_best_fake=False,
    test_all_views=False,
    source_init=False,
    init_region_type="large",
    img_offset=False,
    init_num_pts_per_time=1000,
    init_trbf_c_fix=False,
    init_color_fix_value: float = None,
    **kwargs,
):
    print("Reading Training Transforms...")
    train_json = "transforms_train_hyfluid.json"
    if train_views != "0134" and train_views_fake is None:
        # in this mode, just using some real views, no fake views for fitting
        train_json = f"transforms_train_{train_views}_hyfluid.json"
    train_cam_infos, bbox_model = read_cameras_from_transforms_hyfluid(
        source_path,
        train_json,
        white_background,
        extension,
        start_time,
        duration,
        time_step,
        max_timestamp,
        grey_image,
        train_views,
        train_views_fake,
        use_best_fake,
        img_offset,
    )

    print("Reading Test Transforms...")
    test_json = "transforms_test_hyfluid.json"
    if test_all_views:
        print("Using all views for testing")
        test_json = f"transforms_train_test_hyfluid.json"
    test_cam_infos, _ = read_cameras_from_transforms_hyfluid(
        source_path,
        test_json,
        white_background,
        extension,
        start_time,
        duration,
        time_step,
        max_timestamp,
        grey_image,
        train_views,
        train_views_fake,
        use_best_fake,
        img_offset,
    )

    nerf_normalization = get_nerf_pp_norm(train_cam_infos)

    total_ply_path = os.path.join(model_path, "points3d_total.ply")
    if os.path.exists(total_ply_path):
        os.remove(total_ply_path)

    img_channel = 1 if grey_image else 3

    if init_region_type == "large":
        radius_max = 0.18  # default value 0.18  source region 0.026
        x_mid = 0.34  # default value 0.34 source region 0.34
        y_min = -0.01  # default value -0.01  source region -0.01
        y_max = 0.7  # default value 0.7  source region 0.05
        z_mid = -0.225  # default value -0.225  source region -0.225

    elif init_region_type == "small":
        radius_max = 0.026  # default value 0.18  source region 0.026
        x_mid = 0.34  # default value 0.34 source region 0.34
        y_min = -0.01  # default value -0.01  source region -0.01
        y_max = 0.03  # default value 0.7  source region 0.05
        z_mid = -0.225  # default value -0.225  source region -0.225

    elif init_region_type == "adaptive":
        radius_max_range = [0.026, 0.18]
        x_mid = 0.34
        z_mid = -0.225
        y_min = -0.01
        y_max_range = [0.03, 0.7]

    else:
        raise ValueError(f"Unknown init_region_type: {init_region_type}")

    if source_init:
        num_pts = init_num_pts_per_time
        print(f"Init {num_pts} points with {init_region_type} region type with source_init mode.")
        assert init_region_type in ["small", "large"], f"In source_init mode, init_region_type must be small or large."
        print(f"Generating source_init random point cloud ({num_pts})...")
        y = np.random.uniform(y_min, y_max, (num_pts, 1))  # [-0.05, 0.15] [-0.05, 0.7]

        radius = np.random.random((num_pts, 1)) * radius_max  # * 0.03 # 0.18
        theta = np.random.random((num_pts, 1)) * 2 * np.pi
        x = radius * np.cos(theta) + x_mid
        z = radius * np.sin(theta) + z_mid

        xyz = np.concatenate((x, y, z), axis=1)

        shs = np.random.random((num_pts, img_channel)) / 255.0
        # rgb = np.random.random((num_pts, 3)) * 255.0
        rgb = sh2rgb(shs) * 255

        # print(f"init time {(i - start_time) / duration}")
        # when using our adding source, the time is not directly used
        time = np.zeros((xyz.shape[0], 1))

    else:
        # if the render pipeline is time-based activation and the init_region_type is large, the number of points should be larger
        num_pts = init_num_pts_per_time
        total_xyz = []
        total_rgb = []
        total_time = []
        print(f"Init {num_pts} points per time with {init_region_type} region type with time-based mode.")
        for i in range(start_time, start_time + duration, time_step):
            if init_region_type == "adaptive":
                y_max = y_max_range[0] + (y_max_range[1] - y_max_range[0]) * (i - start_time) / duration
                radius_max = (
                    radius_max_range[0] + (radius_max_range[1] - radius_max_range[0]) * (i - start_time) / duration
                )

            y = np.random.uniform(y_min, y_max, (num_pts, 1))

            radius = np.random.random((num_pts, 1)) * radius_max
            theta = np.random.random((num_pts, 1)) * 2 * np.pi
            x = radius * np.cos(theta) + x_mid
            z = radius * np.sin(theta) + z_mid

            # print(f"Points init x: {x.min()}, {x.max()}")
            # print(f"Points init y: {y.min()}, {y.max()}")
            # print(f"Points init z: {z.min()}, {z.max()}")

            xyz = np.concatenate((x, y, z), axis=1)

            if init_color_fix_value is not None and isinstance(init_color_fix_value, float):
                # 0.6 does not matter, the init value in Gaussian Model is used
                rgb = np.ones((num_pts, img_channel)) * init_color_fix_value * 255.0
            else:
                shs = np.random.random((num_pts, img_channel)) / 255.0
                rgb = np.random.random((num_pts, 3)) * 255.0
                rgb = sh2rgb(shs) * 255

            total_xyz.append(xyz)
            # rgb is not used for fixed color
            total_rgb.append(rgb)
            # print(f"init time {(i - start_time) / duration}")
            # when using our adding source, the time is not directly used
            if init_trbf_c_fix:
                total_time.append(np.zeros((xyz.shape[0], 1)))
            else:
                total_time.append(np.ones((xyz.shape[0], 1)) * (i - start_time) / duration * max_timestamp)

        xyz = np.concatenate(total_xyz, axis=0)
        rgb = np.concatenate(total_rgb, axis=0)
        time = np.concatenate(total_time, axis=0)

    assert xyz.shape[0] == rgb.shape[0]

    xyzt = np.concatenate((xyz, time), axis=1)
    store_ply(total_ply_path, xyzt, rgb, grey_image)

    try:
        pcd = fetch_ply(total_ply_path, grey_image)
    except:
        pcd = None

    assert pcd is not None, "Point cloud could not be loaded!"

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=total_ply_path,
        bbox_model=bbox_model,
    )
    return scene_info


def read_nerf_synthetic_info_hyfluid_valid(
    source_path,
    model_path,
    white_background,
    eval,
    extension=".png",
    start_time=50,
    duration=50,
    time_step=1,
    max_timestamp=1.0,
    grey_image=False,
    train_views="0134",
    train_views_fake=None,
    use_best_fake=False,
    test_all_views=False,
    img_offset=False,
    **kwargs,
):

    print("Reading Test Transforms...")
    test_json = "transforms_test_hyfluid.json"
    if test_all_views:
        print("Using all views for testing")
        test_json = f"transforms_train_test_hyfluid.json"
    test_cam_infos, bbox_model = read_cameras_from_transforms_hyfluid(
        source_path,
        test_json,
        white_background,
        extension,
        start_time,
        duration,
        time_step,
        max_timestamp,
        grey_image,
        train_views,
        train_views_fake,
        use_best_fake,
        img_offset,
    )

    nerf_normalization = get_nerf_pp_norm(test_cam_infos)

    total_ply_path = os.path.join(model_path, "points3d_total.ply")
    pcd = fetch_ply(total_ply_path, grey_image)

    assert pcd is not None, "Point cloud could not be loaded!"

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=test_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=total_ply_path,
        bbox_model=bbox_model,
    )
    return scene_info


def read_nerf_synthetic_info_syn_particle(
    source_path,
    model_path,
    white_background,
    eval,
    extension=".png",
    start_time=50,
    duration=50,
    time_step=1,
    max_timestamp=1.0,
    grey_image=False,
    train_views="0134",
    train_views_fake=None,
    use_best_fake=False,
    test_all_views=False,
    **kwargs,
):
    print("Reading Training Transforms...")
    train_json = "transforms_train_hyfluid.json"
    if train_views != "0134" and train_views_fake is None:
        # in this mode, just using some real views, no fake views for fitting
        train_json = f"transforms_train_{train_views}_hyfluid.json"
    train_cam_infos, bbox_model = read_cameras_from_transforms_hyfluid(
        source_path,
        train_json,
        white_background,
        extension,
        start_time,
        duration,
        time_step,
        max_timestamp,
        grey_image,
        train_views,
        train_views_fake,
        use_best_fake,
        img_offset=False,
    )

    print("Reading Test Transforms...")
    test_json = "transforms_test_hyfluid.json"
    if test_all_views:
        print("Using all views for testing")
        test_json = f"transforms_train_test_hyfluid.json"
    test_cam_infos, _ = read_cameras_from_transforms_hyfluid(
        source_path,
        test_json,
        white_background,
        extension,
        start_time,
        duration,
        time_step,
        max_timestamp,
        grey_image,
        train_views,
        train_views_fake,
        use_best_fake,
        img_offset=False,
    )

    nerf_normalization = get_nerf_pp_norm(train_cam_infos)

    total_ply_path = os.path.join(model_path, "points3d_total.ply")
    if os.path.exists(total_ply_path):
        os.remove(total_ply_path)

    img_channel = 1 if grey_image else 3

    init_region_type = "small"

    if init_region_type == "large":
        radius_max = 0.18  # default value 0.18  source region 0.026
        x_mid = 0.34  # default value 0.34 source region 0.34
        y_min = -0.01  # default value -0.01  source region -0.01
        y_max = 0.7  # default value 0.7  source region 0.05
        z_mid = -0.225  # default value -0.225  source region -0.225

    elif init_region_type == "small":
        radius_max = 0.026  # default value 0.18  source region 0.026
        x_mid = 0.34  # default value 0.34 source region 0.34
        y_min = -0.01  # default value -0.01  source region -0.01
        y_max = 0.03  # default value 0.7  source region 0.05
        z_mid = -0.225  # default value -0.225  source region -0.225

    elif init_region_type == "adaptive":
        radius_max_range = [0.026, 0.18]
        x_mid = 0.34
        z_mid = -0.225
        y_min = -0.01
        y_max_range = [0.03, 0.7]

    else:
        raise ValueError(f"Unknown init_region_type: {init_region_type}")

    # if the render pipeline is time-based activation and the init_region_type is large, the number of points should be larger
    num_pts = 1
    total_xyz = []
    total_rgb = []
    total_time = []
    print(f"Init {num_pts} points per time with {init_region_type} region type with time-based mode.")
    for i in range(start_time, start_time + duration, time_step):
        if i >= 10:
            break
        if init_region_type == "adaptive":
            y_max = y_max_range[0] + (y_max_range[1] - y_max_range[0]) * (i - start_time) / duration
            radius_max = (
                radius_max_range[0] + (radius_max_range[1] - radius_max_range[0]) * (i - start_time) / duration
            )

        y = np.random.uniform(y_min, y_max, (num_pts, 1))

        radius = np.random.random((num_pts, 1)) * radius_max
        theta = np.random.random((num_pts, 1)) * 2 * np.pi
        x = radius * np.cos(theta) + x_mid
        z = radius * np.sin(theta) + z_mid

        # print(f"Points init x: {x.min()}, {x.max()}")
        # print(f"Points init y: {y.min()}, {y.max()}")
        # print(f"Points init z: {z.min()}, {z.max()}")

        xyz = np.concatenate((x, y, z), axis=1)

        # shs = np.random.random((num_pts, img_channel)) / 255.0
        # rgb = np.random.random((num_pts, 3)) * 255.0
        # rgb = sh2rgb(shs) * 255

        # 0.6 does not matter, the init value in Gaussian Model is used
        rgb = np.ones((num_pts, img_channel)) * 0.6 * 255.0

        total_xyz.append(xyz)
        # rgb is not used for fixed color
        total_rgb.append(rgb)
        # print(f"init time {(i - start_time) / duration}")
        # when using our adding source, the time is not directly used
        # total_time.append(np.ones((xyz.shape[0], 1)) * (i - start_time) / duration)
        total_time.append(np.zeros((xyz.shape[0], 1)))

        xyz = np.concatenate(total_xyz, axis=0)
        rgb = np.concatenate(total_rgb, axis=0)
        time = np.concatenate(total_time, axis=0)

    assert xyz.shape[0] == rgb.shape[0]

    xyzt = np.concatenate((xyz, time), axis=1)
    store_ply(total_ply_path, xyzt, rgb, grey_image)

    try:
        pcd = fetch_ply(total_ply_path, grey_image)
    except:
        pcd = None

    assert pcd is not None, "Point cloud could not be loaded!"

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=total_ply_path,
        bbox_model=bbox_model,
    )
    return scene_info


def read_nerf_synthetic_info_syn_particle_valid(
    source_path,
    model_path,
    white_background,
    eval,
    extension=".png",
    start_time=50,
    duration=50,
    time_step=1,
    max_timestamp=1.0,
    grey_image=False,
    train_views="0134",
    train_views_fake=None,
    use_best_fake=False,
    test_all_views=False,
    img_offset=False,
    **kwargs,
):

    print("Reading Test Transforms...")
    test_json = "transforms_test_hyfluid.json"
    if test_all_views:
        print("Using all views for testing")
        test_json = f"transforms_train_test_hyfluid.json"
    test_cam_infos, bbox_model = read_cameras_from_transforms_hyfluid(
        source_path,
        test_json,
        white_background,
        extension,
        start_time,
        duration,
        time_step,
        max_timestamp,
        grey_image,
        train_views,
        train_views_fake,
        use_best_fake,
        img_offset=False,
    )

    nerf_normalization = get_nerf_pp_norm(test_cam_infos)

    total_ply_path = os.path.join(model_path, "points3d_total.ply")
    pcd = fetch_ply(total_ply_path, grey_image)

    assert pcd is not None, "Point cloud could not be loaded!"

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=test_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=total_ply_path,
        bbox_model=bbox_model,
    )
    return scene_info


scene_load_type_callbacks = {
    "hyfluid": read_nerf_synthetic_info_hyfluid,
    "hyfluid_valid": read_nerf_synthetic_info_hyfluid_valid,
    "synthetic_particle": read_nerf_synthetic_info_syn_particle,
    "synthetic_particle_valid": read_nerf_synthetic_info_syn_particle_valid,
}
