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

import json
import os

import torch

from gaussian_splatting.arguments import ModelParams
from gaussian_splatting.gaussian.gaussian_model import GaussianModel
from gaussian_splatting.scene.dataset_readers import scene_load_type_callbacks
from gaussian_splatting.utils.camera_utils import (
    camera_list_from_cam_infos_v2,
    camera_to_json,
)
from gaussian_splatting.utils.system_utils import search_for_max_iteration
from helper_train import record_points_helper


class Scene:

    def __init__(
        self,
        args: ModelParams,
        gaussians: GaussianModel,
        load_iteration=None,
        shuffle=True,
        resolution_scales=[1.0],
        multi_view=False,
        loader="colmap",
        test_all_views=False,
    ):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.ref_model_path = None

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = search_for_max_iteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print(f"Loading trained model at iteration {self.loaded_iter}")

        self.train_cameras = {}
        self.test_cameras = {}
        ray_dict = {}

        if loader in ["hyfluid", "hyfluid_valid", "synthetic_particle", "synthetic_particle_valid"]:
            scene_info = scene_load_type_callbacks[loader](
                args.source_path,
                args.model_path,
                args.white_background,
                args.eval,
                start_time=args.start_time,
                duration=args.duration,
                time_step=args.time_step,
                max_timestamp=args.max_timestamp,
                grey_image=args.grey_image,
                train_views=args.train_views,
                train_views_fake=args.train_views_fake,
                use_best_fake=args.use_best_fake,
                test_all_views=test_all_views,
                source_init=args.source_init,
                init_region_type=args.init_region_type,
                img_offset=args.img_offset,
                init_num_pts_per_time=args.init_num_pts_per_time,
                init_trbf_c_fix=args.init_trbf_c_fix,
                init_color_fix_value=args.init_color_fix_value,
            )

        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, "rb") as src_file, open(
                os.path.join(self.model_path, "input.ply"), "wb"
            ) as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            cam_list = []
            if scene_info.test_cameras:
                cam_list.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                cam_list.extend(scene_info.train_cameras)
            for id, cam in enumerate(cam_list):
                json_cams.append(camera_to_json(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), "w") as file:
                json.dump(json_cams, file, indent=2)

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        self.bbox_model = scene_info.bbox_model

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            if loader in [
                "hyfluid_valid",
                "synthetic_particle_valid",
            ]:
                self.train_cameras[resolution_scale] = []  # no training data

            else:
                self.train_cameras[resolution_scale] = camera_list_from_cam_infos_v2(
                    scene_info.train_cameras, resolution_scale, args
                )

            print("Loading Test Cameras")
            if loader in [
                "hyfluid",
                "hyfluid_valid",
                "synthetic_particle",
                "synthetic_particle_valid",
            ]:
                # we need gt for metrics
                self.test_cameras[resolution_scale] = camera_list_from_cam_infos_v2(
                    scene_info.test_cameras, resolution_scale, args
                )
            else:
                raise NotImplementedError(f"Loader {loader} not implemented")

        for cam in self.train_cameras[resolution_scale]:
            if cam.image_name not in ray_dict and cam.rayo is not None:
                # rays_o, rays_d = 1, camera_direct
                ray_dict[cam.image_name] = torch.cat([cam.rayo, cam.rayd], dim=1).cuda()  # 1 x 6 x H x W

        for cam in self.test_cameras[resolution_scale]:
            if cam.image_name not in ray_dict and cam.rayo is not None:
                ray_dict[cam.image_name] = torch.cat([cam.rayo, cam.rayd], dim=1).cuda()  # 1 x 6 x H x W

        for cam in self.train_cameras[resolution_scale]:
            cam.rays = ray_dict[cam.image_name]

        for cam in self.test_cameras[resolution_scale]:
            cam.rays = ray_dict[cam.image_name]

        if self.loaded_iter:
            self.gaussians.load_ply(
                os.path.join(self.model_path, "point_cloud", f"iteration_{self.loaded_iter}", "point_cloud.ply")
            )
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, f"point_cloud/iteration_{iteration}")
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def sim_save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, f"point_cloud_sim/iteration_{iteration}")
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def record_points(self, iteration, string, two_level=False):
        num_points = self.gaussians._xyz.shape[0]
        record_points_helper(self.model_path, num_points, iteration, string)
        if two_level:
            num_level_1_points = self.gaussians._level_1_parent_idx.shape[0]
            record_points_helper(self.model_path, num_level_1_points, iteration, string + "_level_1")

    def get_train_cameras(self, scale=1.0):
        return self.train_cameras[scale]

    def get_test_cameras(self, scale=1.0):
        return self.test_cameras[scale]
