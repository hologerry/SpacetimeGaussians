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

from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn

from gaussian_splatting.utils.general_utils import (
    build_rotation,
    build_scaling_rotation,
    get_expon_lr_func,
    inverse_sigmoid,
    strip_symmetric,
)
from gaussian_splatting.utils.graphics_utils import BasicPointCloud, pix2ndc
from gaussian_splatting.utils.system_utils import mkdir_p


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        def step_function(delta_t):
            return (delta_t >= 0).float()

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize
        # self.feature_act = torch.sigmoid
        self.t_activation = step_function

    def __init__(self, sh_degree: int, rgb_function="rgbv1"):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree

        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        # self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._motion = torch.empty(0)
        self._omega = torch.empty(0)
        self._trbf_center = torch.empty(0)
        self._trbf_scale = torch.empty(0)
        self._parent_idx = torch.empty(0)  # dummy

        self._delta_rot_radius = torch.empty(0)  # dummy
        self._delta_rot_angle_vel = torch.empty(0)  # dummy

        self._level_1_xyz = torch.empty(0)  # dummy
        self._level_1_features_dc = torch.empty(0)
        self._level_1_scaling = torch.empty(0)
        self._level_1_rotation = torch.empty(0)
        self._level_1_opacity = torch.empty(0)
        self._level_1_omega = torch.empty(0)
        self._level_1_motion = torch.empty(0)
        self._level_1_trbf_center = torch.empty(0)
        self._level_1_trbf_scale = torch.empty(0)
        self._level_1_parent_idx = torch.empty(0)

        self._level_1_delta_rot_radius = torch.empty(0)
        self._level_1_delta_rot_angle_vel = torch.empty(0)

        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.omega_mask = None

        self.level_1_max_radii2D = torch.empty(0)
        self.level_1_xyz_gradient_accum = torch.empty(0)
        self.level_1_denom = torch.empty(0)
        self.level_1_omega_mask = None

        self.percent_dense = 0

        self.optimizer = None
        self.spatial_lr_scale = 0

        self.level_1_optimizer = None
        self.level_1_spatial_lr_scale = 0

        self.rgb_decoder = None

        self.setup_functions()
        self.delta_t = None

        self.mask_for_ems = None
        self.distance_to_camera = None
        self.trbf_scale_init = None
        self.ts = None
        self.trbf_output = None
        self.preprocess_points = False
        self.add_sph_points_scale = 0.8
        self.level_1_trbf_output = None
        self.all_trbf_output = None

        self.max_z, self.min_z = 0.0, 0.0
        self.max_y, self.min_y = 0.0, 0.0
        self.max_x, self.min_x = 0.0, 0.0

        self.computed_trbf_scale = None
        self.computed_opacity = None
        self.computed_scales = None

        self.computed_level_1_trbf_scale = None
        self.computed_level_1_opacity = None
        self.computed_level_1_scales = None

        self.ray_start = 0.7

    # def capture(self):
    #     return (
    #         self.active_sh_degree,
    #         self._xyz,
    #         self._features_dc,
    #         self._scaling,
    #         self._rotation,
    #         self._opacity,
    #         self.max_radii2D,
    #         self.xyz_gradient_accum,
    #         self.denom,
    #         self.optimizer.state_dict(),
    #         self.spatial_lr_scale,
    #     )

    # def restore(self, model_args, training_args):
    #     (
    #         self.active_sh_degree,
    #         self._xyz,
    #         self._features_dc,
    #         self._scaling,
    #         self._rotation,
    #         self._opacity,
    #         self.max_radii2D,
    #         xyz_gradient_accum,
    #         denom,
    #         opt_dict,
    #         self.spatial_lr_scale,
    #     ) = model_args
    #     self.training_setup(training_args)
    #     self.xyz_gradient_accum = xyz_gradient_accum
    #     self.denom = denom
    #     self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_level_1_scaling(self):
        return self.scaling_activation(self._level_1_scaling)

    @property
    def get_all_scaling(self):
        return torch.cat((self.get_scaling, self.get_level_1_scaling), dim=0)

    def get_rotation(self, delta_t):
        n_points = self._xyz.shape[0]
        cur_delta_t = delta_t[:n_points]
        rotation = self._rotation + cur_delta_t * self._omega
        self.delta_t = cur_delta_t
        return self.rotation_activation(rotation)

    def get_level_1_rotation(self, level_1_delta_t):
        level_1_rotation = self._level_1_rotation + level_1_delta_t * self._level_1_omega
        self.level_1_delta_t = level_1_delta_t
        return self.rotation_activation(level_1_rotation)

    def get_all_rotation(self, delta_t):
        return torch.cat((self.get_rotation(delta_t), self.get_level_1_rotation(delta_t)), dim=0)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_level_1_xyz(self):
        n_level_1_points = self._level_1_parent_idx.shape[0]
        self._level_1_xyz = torch.zeros((n_level_1_points, 3), dtype=torch.float, device="cuda")
        return self._level_1_xyz

    @property
    def get_all_xyz(self):
        return torch.cat((self._xyz, self._level_1_xyz), dim=0)

    @property
    def get_level_1_parent_idx(self):
        return self._level_1_parent_idx

    @property
    def get_motion(self):
        return self._motion

    @property
    def get_level_1_motion(self):
        self._level_1_motion = torch.zeros((self._level_1_parent_idx.shape[0], 9), device="cuda")
        return self._level_1_motion

    @property
    def get_all_motion(self):
        return torch.cat((self._motion, self._level_1_motion), dim=0)

    @property
    def get_level_1_delta_rot_radius(self):
        return self._level_1_delta_rot_radius

    @property
    def get_level_1_delta_rot_angle_vel(self):
        return self._level_1_delta_rot_angle_vel

    @property
    def get_trbf_center(self):
        return self._trbf_center

    @property
    def get_level_1_trbf_center(self):
        return self._level_1_trbf_center

    @property
    def get_all_trbf_center(self):
        return torch.cat((self._trbf_center, self._level_1_trbf_center), dim=0)

    @property
    def get_trbf_scale(self):
        return self._trbf_scale

    @property
    def get_level_1_trbf_scale(self):
        return self._level_1_trbf_scale

    @property
    def get_all_trbf_scale(self):
        return torch.cat((self._trbf_scale, self._level_1_trbf_scale), dim=0)

    @property
    def get_features(self):
        return self._features_dc

    @property
    def get_level_1_features(self):
        return self._level_1_features_dc

    @property
    def get_all_features(self):
        return torch.cat((self._features_dc, self._level_1_features_dc), dim=0)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_level_1_opacity(self):
        return self.opacity_activation(self._level_1_opacity)

    @property
    def get_all_opacity(self):
        return torch.cat((self.get_opacity, self.get_level_1_opacity), dim=0)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def get_level_1_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_level_1_scaling, scaling_modifier, self._level_1_rotation)

    def get_all_covariance(self, scaling_modifier=1):
        return torch.cat((self.get_covariance(scaling_modifier), self.get_level_1_covariance(scaling_modifier)), dim=0)

    def one_up_sh_degree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):

        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = torch.tensor(np.asarray(pcd.colors)).float().cuda()
        times = torch.tensor(np.asarray(pcd.times)).float().cuda()

        print("Number of points at initialization : ", fused_point_cloud.shape[0])

        # dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        # scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        # scales = torch.clamp(scales, -10, 1.0)

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1.0

        opacities = inverse_sigmoid(
            0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")
        )

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        print(f"self._xyz inited {self._xyz}")

        self._parent_idx = torch.zeros((fused_point_cloud.shape[0], 1), dtype=torch.long, device="cuda") - 1
        print(f"self._parent_idx inited {self._parent_idx}")
        # features9channel = fused_color
        # lite just use the base color

        fused_color_fix = torch.zeros_like(fused_color) + 0.6
        self._features_dc = nn.Parameter(fused_color_fix.contiguous().requires_grad_(True))
        print(f"self._features_dc inited {self._features_dc}")

        N, _ = fused_color.shape

        scales = torch.zeros((fused_point_cloud.shape[0], 3), device="cuda") - 6.2

        self._scaling = nn.Parameter(scales.requires_grad_(True))
        print(f"self._scaling inited {self._scaling}")

        self._rotation = nn.Parameter(rots.requires_grad_(True))
        print(f"self._rotation inited {self._rotation}")

        # we keep level 0 _omega, for better saving and loading
        omega = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        self._omega = nn.Parameter(omega.requires_grad_(True))
        print(f"self._omega inited {self._omega}")

        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        print(f"self._opacity inited {self._opacity}")

        motion = torch.zeros((fused_point_cloud.shape[0], 9), device="cuda")  # x1,x2,x3, y1,y2,y3, z1,z2,z3
        self._motion = nn.Parameter(motion.requires_grad_(True))
        print(f"self._motion inited {self._motion}")

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        print(f"self.max_radii2D inited {self.max_radii2D}")

        self._trbf_center = nn.Parameter(times.contiguous().requires_grad_(True))
        self._trbf_scale = nn.Parameter(torch.ones((self.get_xyz.shape[0], 1), device="cuda").requires_grad_(True))

        print(f"self._trbf_center inited {self._trbf_center}")
        print(f"self._trbf_scale inited {self._trbf_scale}")

        self._delta_rot_radius = torch.zeros((fused_point_cloud.shape[0], 1), device="cuda")
        self._delta_rot_angle_vel = torch.zeros((fused_point_cloud.shape[0], 1), device="cuda")
        print(f"self._delta_rot_radius inited {self._delta_rot_radius}")
        print(f"self._delta_rot_angle_vel inited {self._delta_rot_angle_vel}")

        if self.trbf_scale_init is not None:
            nn.init.constant_(self._trbf_scale, self.trbf_scale_init)  # too large ?
        else:
            nn.init.constant_(self._trbf_scale, 0)  # too large ?

        # nn.init.constant_(self._omega, 0)
        self.rgb_grd = {}

        self.max_z, self.min_z = torch.amax(self._xyz[:, 2]), torch.amin(self._xyz[:, 2])
        self.max_y, self.min_y = torch.amax(self._xyz[:, 1]), torch.amin(self._xyz[:, 1])
        self.max_x, self.min_x = torch.amax(self._xyz[:, 0]), torch.amin(self._xyz[:, 0])
        self.max_z = min((self.max_z, 200.0))  # some outliers in the n4d datasets..

    def create_another_level(
        self,
        spatial_lr_scale: float,
        new_pts_per_time: int,
        new_pts_init_op=0.1,
        new_pts_init_color=0.6,
        new_pts_init_xyz="parent",
        new_pts_init_xyz_offset=0.0,
        new_pts_init_scale="dist",
        new_pts_init_min_opacity=0.05,
        new_pts_init_delta_rot_radius_scale=5.0,
        new_pts_init_delta_rot_angle_vel_rand=None,
        start_time=0,
        duration=120,
        time_step=1,
        **kwargs,
    ):
        self.level_1_spatial_lr_scale = spatial_lr_scale

        n_level_0_points = self.get_xyz.shape[0]
        # parent_means_3D = self.get_xyz
        level_0_trbf_center = self.get_trbf_center
        # parent_motion = self._motion
        level_0_opacity = self.get_opacity
        point_times = torch.ones((n_level_0_points, 1), dtype=torch.float, requires_grad=False, device="cuda")
        parent_idx = torch.arange(n_level_0_points, dtype=torch.long, requires_grad=False, device="cuda")
        level_1_total_parent_idx = []
        level_1_total_times = []
        for time_i in range(start_time, start_time + duration, time_step):
            cur_time_stamp = (time_i - start_time) / duration
            if new_pts_init_xyz == "parent_couple":
                trbf_distance_offset = cur_time_stamp * point_times - level_0_trbf_center
                # tforpoly = trbf_distance_offset.detach()
                time_coefficient = self.t_activation(trbf_distance_offset)
                # cur_level_means_3D = (
                #     parent_means_3D
                #     + parent_motion[:, 0:3] * tforpoly * time_coefficient
                #     + parent_motion[:, 3:6] * tforpoly * tforpoly * time_coefficient
                #     + parent_motion[:, 6:9] * tforpoly * tforpoly * tforpoly * time_coefficient
                # )
                time_visible_parent_to_select = torch.where(time_coefficient > 0, True, False).squeeze()
                opacity_visible_parent_to_select = (level_0_opacity > new_pts_init_min_opacity).squeeze()
                visible_parent_to_select = torch.logical_and(
                    time_visible_parent_to_select, opacity_visible_parent_to_select
                )

                cur_visible_parent_idx = parent_idx[visible_parent_to_select]
                cur_level_select_idx = torch.randperm(cur_visible_parent_idx.shape[0], device="cuda")[
                    :new_pts_per_time
                ]
                cur_level_selected_parent_idx = cur_visible_parent_idx[cur_level_select_idx]
                cur_level_selected_parent_idx = cur_level_selected_parent_idx.reshape(-1, 1)
            else:
                raise ValueError(f"new_pts_init_xyz {new_pts_init_xyz} not implemented")

            cur_level_time = torch.ones((cur_level_selected_parent_idx.shape[0], 1), device="cuda") * cur_time_stamp
            level_1_total_parent_idx.append(cur_level_selected_parent_idx)
            level_1_total_times.append(cur_level_time)

        level_1_total_parent_idx = torch.cat(level_1_total_parent_idx, dim=0).long().cuda()
        level_1_total_times = torch.cat(level_1_total_times, dim=0).float().cuda()

        self._level_1_parent_idx = level_1_total_parent_idx
        print(f"self._level_1_parent_idx inited {self._level_1_parent_idx}")

        # dummy
        self._level_1_xyz = torch.zeros((level_1_total_parent_idx.shape[0], 3), device="cuda")
        print(f"self._level_1_xyz inited {self._level_1_xyz}")

        level_1_fused_color = (
            torch.zeros((self._level_1_parent_idx.shape[0], self._features_dc.shape[1])).float().cuda()
        )
        level_1_fused_color = level_1_fused_color + new_pts_init_color
        self._level_1_features_dc = nn.Parameter(level_1_fused_color.requires_grad_(True))
        print(f"self._level_1_features_dc inited {self._level_1_features_dc}")

        level_1_scales = torch.zeros((level_1_total_parent_idx.shape[0], 3), device="cuda") + new_pts_init_scale
        self._level_1_scaling = nn.Parameter(level_1_scales.requires_grad_(True))
        print(f"self._level_1_scaling inited {self._level_1_scaling}")

        level_1_rots = torch.zeros((level_1_total_parent_idx.shape[0], 4), device="cuda")
        level_1_rots[:, 0] = 1.0
        self._level_1_rotation = nn.Parameter(level_1_rots.requires_grad_(True))
        print(f"self._level_1_rotation inited {self._level_1_rotation}")

        level_1_omega = torch.zeros((level_1_total_parent_idx.shape[0], 4), device="cuda")
        self._level_1_omega = nn.Parameter(level_1_omega.requires_grad_(True))
        print(f"self._level_1_omega inited {self._level_1_omega}")

        level_1_opacities = inverse_sigmoid(
            new_pts_init_op * torch.ones((level_1_total_parent_idx.shape[0], 1), dtype=torch.float, device="cuda")
        )
        self._level_1_opacity = nn.Parameter(level_1_opacities.requires_grad_(True))
        print(f"self._level_1_opacity inited {self._level_1_opacity}")

        level_1_motion = torch.zeros((level_1_total_parent_idx.shape[0], 9), device="cuda")
        self._level_1_motion = level_1_motion  # nn.Parameter(level_1_motion.requires_grad_(True))
        print(f"self._level_1_motion inited {self._level_1_motion}")

        self.level_1_max_radii2D = torch.zeros((level_1_total_parent_idx.shape[0]), device="cuda")
        print(f"self.level_1_max_radii2D inited {self.level_1_max_radii2D}")

        level_1_trbf_center = level_1_total_times
        self._level_1_trbf_center = nn.Parameter(level_1_trbf_center.requires_grad_(True))
        print(f"self._level_1_trbf_center inited {self._level_1_trbf_center}")

        level_1_trbf_scale = torch.ones((level_1_total_parent_idx.shape[0], 1), device="cuda")
        self._level_1_trbf_scale = nn.Parameter(level_1_trbf_scale.requires_grad_(True))
        print(f"self._level_1_trbf_scale inited {self._level_1_trbf_scale}")

        level_1_scales_mean = torch.exp(torch.mean(level_1_scales, dim=1, keepdim=True))
        level_1_delta_rot_radius = (
            torch.zeros((level_1_total_parent_idx.shape[0], 1), device="cuda")
            + new_pts_init_delta_rot_radius_scale * level_1_scales_mean
        )
        self._level_1_delta_rot_radius = nn.Parameter(level_1_delta_rot_radius.requires_grad_(True))
        print(f"self._level_1_delta_rot_radius inited {self._level_1_delta_rot_radius}")

        if new_pts_init_delta_rot_angle_vel_rand is not None:
            level_1_delta_rot_angle_vel = (
                torch.rand((level_1_total_parent_idx.shape[0], 1), device="cuda")
                * new_pts_init_delta_rot_angle_vel_rand
            )
        else:
            level_1_delta_rot_angle_vel = torch.zeros((level_1_total_parent_idx.shape[0], 1), device="cuda")
        self._level_1_delta_rot_angle_vel = nn.Parameter(level_1_delta_rot_angle_vel.requires_grad_(True))
        print(f"self._level_1_delta_rot_angle_vel inited {self._level_1_delta_rot_angle_vel}")

    def cache_gradient(self):
        self._xyz_grd += self._xyz.grad.clone()
        self._features_dc_grd += self._features_dc.grad.clone()
        self._scaling_grd += self._scaling.grad.clone()
        self._rotation_grd += self._rotation.grad.clone()
        self._opacity_grd += self._opacity.grad.clone()
        self._trbf_center_grd += self._trbf_center.grad.clone()
        self._trbf_scale_grd += self._trbf_scale.grad.clone()
        self._motion_grd += self._motion.grad.clone()
        self._omega_grd += self._omega.grad.clone()

    def cache_level_1_gradient(self):
        self._level_1_features_dc_grad += self._level_1_features_dc.grad.clone()
        self._level_1_scaling_grad += self._level_1_scaling.grad.clone()
        self._level_1_rotation_grad += self._level_1_rotation.grad.clone()
        self._level_1_opacity_grad += self._level_1_opacity.grad.clone()
        self._level_1_trbf_center_grad += self._level_1_trbf_center.grad.clone()
        self._level_1_trbf_scale_grad += self._level_1_trbf_scale.grad.clone()
        # self._level_1_motion_grad += self._level_1_motion.grad.clone()
        self._level_1_omega_grad += self._level_1_omega.grad.clone()
        self._level_1_delta_rot_radius_grad += self._level_1_delta_rot_radius.grad.clone()
        self._level_1_delta_rot_angle_vel_grad += self._level_1_delta_rot_angle_vel.grad.clone()

    def zero_gradient_cache(self):
        self._xyz_grd = torch.zeros_like(self._xyz, requires_grad=False)
        self._features_dc_grd = torch.zeros_like(self._features_dc, requires_grad=False)
        self._scaling_grd = torch.zeros_like(self._scaling, requires_grad=False)
        self._rotation_grd = torch.zeros_like(self._rotation, requires_grad=False)
        self._opacity_grd = torch.zeros_like(self._opacity, requires_grad=False)
        self._trbf_center_grd = torch.zeros_like(self._trbf_center, requires_grad=False)
        self._trbf_scale_grd = torch.zeros_like(self._trbf_scale, requires_grad=False)
        self._motion_grd = torch.zeros_like(self._motion, requires_grad=False)
        self._omega_grd = torch.zeros_like(self._omega, requires_grad=False)

    def zero_level_1_gradient_cache(self):
        self._level_1_features_dc_grad = torch.zeros_like(self._level_1_features_dc, requires_grad=False)
        self._level_1_scaling_grad = torch.zeros_like(self._level_1_scaling, requires_grad=False)
        self._level_1_rotation_grad = torch.zeros_like(self._level_1_rotation, requires_grad=False)
        self._level_1_opacity_grad = torch.zeros_like(self._level_1_opacity, requires_grad=False)
        self._level_1_trbf_center_grad = torch.zeros_like(self._level_1_trbf_center, requires_grad=False)
        self._level_1_trbf_scale_grad = torch.zeros_like(self._level_1_trbf_scale, requires_grad=False)
        # self._level_1_motion_grad = torch.zeros_like(self._level_1_motion, requires_grad=False)
        self._level_1_omega_grad = torch.zeros_like(self._level_1_omega, requires_grad=False)
        self._level_1_delta_rot_radius_grad = torch.zeros_like(self._level_1_delta_rot_radius, requires_grad=False)
        self._level_1_delta_rot_angle_vel_grad = torch.zeros_like(
            self._level_1_delta_rot_angle_vel, requires_grad=False
        )

    def set_batch_gradient(self, batch_size):
        ratio = 1 / batch_size
        self._xyz.grad = self._xyz_grd * ratio
        self._features_dc.grad = self._features_dc_grd * ratio
        self._scaling.grad = self._scaling_grd * ratio
        self._rotation.grad = self._rotation_grd * ratio
        self._opacity.grad = self._opacity_grd * ratio
        self._trbf_center.grad = self._trbf_center_grd * ratio
        self._trbf_scale.grad = self._trbf_scale_grd * ratio
        self._motion.grad = self._motion_grd * ratio
        self._omega.grad = self._omega_grd * ratio

    def set_level_1_batch_gradient(self, batch_size):
        ratio = 1 / batch_size
        self._level_1_features_dc.grad = self._level_1_features_dc_grad * ratio
        self._level_1_scaling.grad = self._level_1_scaling_grad * ratio
        self._level_1_rotation.grad = self._level_1_rotation_grad * ratio
        self._level_1_opacity.grad = self._level_1_opacity_grad * ratio
        self._level_1_trbf_center.grad = self._level_1_trbf_center_grad * ratio
        self._level_1_trbf_scale.grad = self._level_1_trbf_scale_grad * ratio
        # self._level_1_motion.grad = self._level_1_motion_grad * ratio
        self._level_1_omega.grad = self._level_1_omega_grad * ratio
        self._level_1_delta_rot_radius.grad = self._level_1_delta_rot_radius_grad * ratio
        self._level_1_delta_rot_angle_vel.grad = self._level_1_delta_rot_angle_vel_grad * ratio

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        l = [
            {"params": [self._xyz], "lr": 0.0 * self.spatial_lr_scale, "name": "xyz"},
            {"params": [self._features_dc], "lr": training_args.feature_lr, "name": "f_dc"},
            {"params": [self._opacity], "lr": training_args.opacity_lr, "name": "opacity"},
            {"params": [self._scaling], "lr": training_args.scaling_lr, "name": "scaling"},
            {"params": [self._rotation], "lr": training_args.rotation_lr, "name": "rotation"},
            {"params": [self._omega], "lr": training_args.omega_lr, "name": "omega"},
            {"params": [self._trbf_center], "lr": training_args.trbf_c_lr, "name": "trbf_center"},
            {"params": [self._trbf_scale], "lr": training_args.trbf_s_lr, "name": "trbf_scale"},
            {
                "params": [self._motion],
                "lr": training_args.position_lr_init * self.spatial_lr_scale * 0.5 * training_args.move_lr,
                "name": "motion",
            },
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

    def level_1_training_setup(self, training_args):
        self.level_1_xyz_gradient_accum = torch.zeros((self.get_level_1_parent_idx.shape[0], 1), device="cuda")
        self.level_1_denom = torch.zeros((self.get_level_1_parent_idx.shape[0], 1), device="cuda")
        l = [
            {"params": [self._level_1_features_dc], "lr": training_args.level_1_feature_lr, "name": "f_dc"},
            {"params": [self._level_1_opacity], "lr": training_args.level_1_opacity_lr, "name": "opacity"},
            {"params": [self._level_1_scaling], "lr": training_args.level_1_scaling_lr, "name": "scaling"},
            {"params": [self._level_1_rotation], "lr": training_args.level_1_rotation_lr, "name": "rotation"},
            {"params": [self._level_1_omega], "lr": training_args.level_1_omega_lr, "name": "omega"},
            {"params": [self._level_1_trbf_center], "lr": training_args.level_1_trbf_c_lr, "name": "trbf_center"},
            {"params": [self._level_1_trbf_scale], "lr": training_args.level_1_trbf_s_lr, "name": "trbf_scale"},
            # {
            #     "params": [self._level_1_motion],
            #     "lr": training_args.level_1_position_lr_init
            #     * self.level_1_spatial_lr_scale
            #     * 0.5
            #     * training_args.level_1_move_lr,
            #     "name": "motion",
            # },
            {
                "params": [self._level_1_delta_rot_radius],
                "lr": training_args.level_1_delta_rot_radius_lr,
                "name": "delta_rot_radius",
            },
            {
                "params": [self._level_1_delta_rot_angle_vel],
                "lr": training_args.level_1_delta_rot_angle_vel_lr,
                "name": "delta_rot_angle_vel",
            },
        ]
        self.level_1_optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.level_1_xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.level_1_position_lr_init * self.level_1_spatial_lr_scale,
            lr_final=training_args.level_1_position_lr_final * self.level_1_spatial_lr_scale,
            lr_delay_mult=training_args.level_1_position_lr_delay_mult,
            max_steps=training_args.level_1_position_lr_max_steps,
        )

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                # lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = 0.0
                return 0.0

    def update_level_1_learning_rate(self, iteration):
        """Learning rate scheduling per step"""

        return 0.0
        # since we dont have level_1_xyz, we dont need to update it
        for param_group in self.level_1_optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.level_1_xyz_scheduler_args(iteration)
                param_group["lr"] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "trbf_center", "trbf_scale", "nx", "ny", "nz"]
        # All channels except the 3 DC
        # for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
        #     l.append(f'f_dc_{i}')
        # for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
        #     l.append(f'f_rest_{i}')
        for i in range(self._motion.shape[1]):
            l.append(f"motion_{i}")

        for i in range(self._features_dc.shape[1]):
            l.append(f"f_dc_{i}")
        # for i in range(self._features_rest.shape[1]):
        #     l.append('f_rest_{i}')
        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append(f"scale_{i}")
        for i in range(self._rotation.shape[1]):
            l.append(f"rot_{i}")
        for i in range(self._omega.shape[1]):
            l.append(f"omega_{i}")

        l.append("level")

        l.append("parent_idx")

        l.append("delta_rot_radius")
        l.append("delta_rot_angle_vel")

        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().cpu().numpy()
        # f_rest = self._features_rest.detach().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        trbf_center = self._trbf_center.detach().cpu().numpy()
        trbf_scale = self._trbf_scale.detach().cpu().numpy()
        motion = self._motion.detach().cpu().numpy()
        omega = self._omega.detach().cpu().numpy()
        level = np.zeros((xyz.shape[0], 1))
        parent_idx_dummy = np.zeros((xyz.shape[0], 1)) - 1
        delta_rot_radius_dummy = np.zeros((xyz.shape[0], 1))
        delta_rot_angle_vel_dummy = np.zeros((xyz.shape[0], 1))

        if self._level_1_parent_idx.shape[0] > 0:

            level_1_xyz_dummy = np.zeros((self._level_1_parent_idx.shape[0], 3))
            level_1_normals = np.zeros_like(level_1_xyz_dummy)
            level_1_f_dc = self._level_1_features_dc.detach().cpu().numpy()
            # level_1_f_rest = self._level_1_features_rest.detach().cpu().numpy()
            level_1_opacities = self._level_1_opacity.detach().cpu().numpy()
            level_1_scale = self._level_1_scaling.detach().cpu().numpy()
            level_1_rotation = self._level_1_rotation.detach().cpu().numpy()
            level_1_trbf_center = self._level_1_trbf_center.detach().cpu().numpy()
            level_1_trbf_scale = self._level_1_trbf_scale.detach().cpu().numpy()
            level_1_motion_dummy = np.zeros((self._level_1_parent_idx.shape[0], 9))
            level_1_omega = self._level_1_omega.detach().cpu().numpy()
            level_1_level = np.ones((self._level_1_parent_idx.shape[0], 1))
            level_1_parent_idx = self._level_1_parent_idx.detach().cpu().numpy()

            level_1_delta_rot_radius = self._level_1_delta_rot_radius.detach().cpu().numpy()
            level_1_delta_rot_angle_vel = self._level_1_delta_rot_angle_vel.detach().cpu().numpy()

            all_xyz = np.concatenate((xyz, level_1_xyz_dummy), axis=0)
            all_normals = np.concatenate((normals, level_1_normals), axis=0)
            all_f_dc = np.concatenate((f_dc, level_1_f_dc), axis=0)
            # all_f_rest = np.concatenate((f_rest, level_1_f_rest), axis=0)
            all_opacities = np.concatenate((opacities, level_1_opacities), axis=0)
            all_scale = np.concatenate((scale, level_1_scale), axis=0)
            all_rotation = np.concatenate((rotation, level_1_rotation), axis=0)
            all_trbf_center = np.concatenate((trbf_center, level_1_trbf_center), axis=0)
            all_trbf_scale = np.concatenate((trbf_scale, level_1_trbf_scale), axis=0)
            all_motion = np.concatenate((motion, level_1_motion_dummy), axis=0)
            all_omega = np.concatenate((omega, level_1_omega), axis=0)
            all_level = np.concatenate((level, level_1_level), axis=0)
            all_parent_idx = np.concatenate((parent_idx_dummy, level_1_parent_idx), axis=0)

            all_delta_rot_radius = np.concatenate((delta_rot_radius_dummy, level_1_delta_rot_radius), axis=0)
            all_delta_rot_angle_vel = np.concatenate((delta_rot_angle_vel_dummy, level_1_delta_rot_angle_vel), axis=0)

        else:
            all_xyz = xyz
            all_normals = normals
            all_f_dc = f_dc
            # all_f_rest = f_rest
            all_opacities = opacities
            all_scale = scale
            all_rotation = rotation
            all_trbf_center = trbf_center
            all_trbf_scale = trbf_scale
            all_motion = motion
            all_omega = omega
            all_level = level
            all_parent_idx = parent_idx_dummy

            all_delta_rot_radius = delta_rot_radius_dummy
            all_delta_rot_angle_vel = delta_rot_angle_vel_dummy

        dtype_full = [(attribute, "f4") for attribute in self.construct_list_of_attributes()]

        elements = np.empty(all_xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (
                all_xyz,
                all_trbf_center,
                all_trbf_scale,
                all_normals,
                all_motion,
                all_f_dc,
                all_opacities,
                all_scale,
                all_rotation,
                all_omega,
                all_level,
                all_parent_idx,
                all_delta_rot_radius,
                all_delta_rot_angle_vel,
            ),
            axis=1,
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

        txt_fname = path.replace(".ply", ".txt")
        np.savetxt(txt_fname, attributes, delimiter=",")

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        ply_data = PlyData.read(path)

        xyz = np.stack(
            (
                np.asarray(ply_data.elements[0]["x"]),
                np.asarray(ply_data.elements[0]["y"]),
                np.asarray(ply_data.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(ply_data.elements[0]["opacity"])[..., np.newaxis]

        trbf_center = np.asarray(ply_data.elements[0]["trbf_center"])[..., np.newaxis]
        trbf_scale = np.asarray(ply_data.elements[0]["trbf_scale"])[..., np.newaxis]

        # motion = np.asarray(ply_data.elements[0]["motion"])
        motion_names = [p.name for p in ply_data.elements[0].properties if p.name.startswith("motion")]
        num_motion = 9
        motion = np.zeros((xyz.shape[0], num_motion))
        for i in range(num_motion):
            motion[:, i] = np.asarray(ply_data.elements[0]["motion_" + str(i)])

        dc_f_names = [p.name for p in ply_data.elements[0].properties if p.name.startswith("f_dc")]
        num_dc_features = len(dc_f_names)

        features_dc = np.zeros((xyz.shape[0], num_dc_features))
        for i in range(num_dc_features):
            features_dc[:, i] = np.asarray(ply_data.elements[0]["f_dc_" + str(i)])

        extra_f_names = [p.name for p in ply_data.elements[0].properties if p.name.startswith("f_rest_")]
        # assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(ply_data.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], -1))

        scale_names = [p.name for p in ply_data.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(ply_data.elements[0][attr_name])

        rot_names = [p.name for p in ply_data.elements[0].properties if p.name.startswith("rot")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(ply_data.elements[0][attr_name])

        omega_names = [p.name for p in ply_data.elements[0].properties if p.name.startswith("omega")]
        omegas = np.zeros((xyz.shape[0], len(omega_names)))
        for idx, attr_name in enumerate(omega_names):
            omegas[:, idx] = np.asarray(ply_data.elements[0][attr_name])

        # ft_names = [p.name for p in ply_data.elements[0].properties if p.name.startswith("f_t")]
        # ft_omegas = np.zeros((xyz.shape[0], len(ft_names)))
        # for idx, attr_name in enumerate(ft_names):
        #     ft_omegas[:, idx] = np.asarray(ply_data.elements[0][attr_name])

        level = np.asarray(ply_data.elements[0]["level"])[..., np.newaxis]
        level_0_idx = (level == 0).squeeze()
        level_1_idx = (level == 1).squeeze()

        parent_idx = np.asarray(ply_data.elements[0]["parent_idx"])[..., np.newaxis]

        delta_rot_radius = np.asarray(ply_data.elements[0]["delta_rot_radius"])[..., np.newaxis]
        delta_rot_angle_vel = np.asarray(ply_data.elements[0]["delta_rot_angle_vel"])[..., np.newaxis]

        self._xyz = nn.Parameter(torch.tensor(xyz[level_0_idx], dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc[level_0_idx], dtype=torch.float, device="cuda").requires_grad_(True)
        )
        # self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(
            torch.tensor(opacities[level_0_idx], dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._scaling = nn.Parameter(
            torch.tensor(scales[level_0_idx], dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(rots[level_0_idx], dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._trbf_center = nn.Parameter(
            torch.tensor(trbf_center[level_0_idx], dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._trbf_scale = nn.Parameter(
            torch.tensor(trbf_scale[level_0_idx], dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._motion = nn.Parameter(
            torch.tensor(motion[level_0_idx], dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._omega = nn.Parameter(
            torch.tensor(omegas[level_0_idx], dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._parent_idx = torch.tensor(parent_idx[level_0_idx], dtype=torch.long, device="cuda")
        self._delta_rot_radius = torch.tensor(delta_rot_radius[level_0_idx], dtype=torch.float, device="cuda")
        self._delta_rot_angle_vel = torch.tensor(delta_rot_angle_vel[level_0_idx], dtype=torch.float, device="cuda")

        self.active_sh_degree = self.max_sh_degree
        self.computed_trbf_scale = torch.exp(self._trbf_scale)  # precomputed
        self.computed_opacity = self.opacity_activation(self._opacity)
        self.computed_scales = torch.exp(self._scaling)  # change not very large

        if np.sum(level_1_idx) > 0:
            self._level_1_parent_idx = torch.tensor(parent_idx[level_1_idx], dtype=torch.long, device="cuda")
            self._level_1_xyz = torch.tensor(xyz[level_1_idx], dtype=torch.float, device="cuda")
            self._level_1_features_dc = nn.Parameter(
                torch.tensor(features_dc[level_1_idx], dtype=torch.float, device="cuda").requires_grad_(True)
            )
            self._level_1_opacity = nn.Parameter(
                torch.tensor(opacities[level_1_idx], dtype=torch.float, device="cuda").requires_grad_(True)
            )
            self._level_1_scaling = nn.Parameter(
                torch.tensor(scales[level_1_idx], dtype=torch.float, device="cuda").requires_grad_(True)
            )
            self._level_1_rotation = nn.Parameter(
                torch.tensor(rots[level_1_idx], dtype=torch.float, device="cuda").requires_grad_(True)
            )
            self._level_1_trbf_center = nn.Parameter(
                torch.tensor(trbf_center[level_1_idx], dtype=torch.float, device="cuda").requires_grad_(True)
            )
            self._level_1_trbf_scale = nn.Parameter(
                torch.tensor(trbf_scale[level_1_idx], dtype=torch.float, device="cuda").requires_grad_(True)
            )
            self._level_1_motion = torch.tensor(motion[level_1_idx], dtype=torch.float, device="cuda")

            self._level_1_omega = nn.Parameter(
                torch.tensor(omegas[level_1_idx], dtype=torch.float, device="cuda").requires_grad_(True)
            )

            self._level_1_delta_rot_radius = nn.Parameter(
                torch.tensor(delta_rot_radius[level_1_idx], dtype=torch.float, device="cuda").requires_grad_(True)
            )
            self._level_1_delta_rot_angle_vel = nn.Parameter(
                torch.tensor(delta_rot_angle_vel[level_1_idx], dtype=torch.float, device="cuda").requires_grad_(True)
            )

            self.computed_level_1_trbf_scale = torch.exp(self._level_1_trbf_scale)  # precomputed
            self.computed_level_1_opacity = self.opacity_activation(self._level_1_opacity)
            self.computed_level_1_scales = torch.exp(self._level_1_scaling)  # change not very large

            self.computed_all_trbf_scale = torch.cat(
                (self.computed_trbf_scale, self.computed_level_1_trbf_scale), dim=0
            )
            self.computed_all_opacity = torch.cat((self.computed_opacity, self.computed_level_1_opacity), dim=0)
            self.computed_all_scales = torch.cat((self.computed_scales, self.computed_level_1_scales), dim=0)

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)

                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"]) == 1:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                    del self.optimizer.state[group["params"][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                    self.optimizer.state[group["params"][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_level_1_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.level_1_optimizer.param_groups:
            if len(group["params"]) == 1:
                stored_state = self.level_1_optimizer.state.get(group["params"][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                    del self.level_1_optimizer.state[group["params"][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                    self.level_1_optimizer.state[group["params"][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        # self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._trbf_center = optimizable_tensors["trbf_center"]
        self._trbf_scale = optimizable_tensors["trbf_scale"]
        self._motion = optimizable_tensors["motion"]
        self._omega = optimizable_tensors["omega"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        if self.omega_mask is not None:
            self.omega_mask = self.omega_mask[valid_points_mask]

    def prune_points_level_1(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_level_1_optimizer(valid_points_mask)
        self._level_1_features_dc = optimizable_tensors["f_dc"]
        self._level_1_opacity = optimizable_tensors["opacity"]
        self._level_1_scaling = optimizable_tensors["scaling"]
        self._level_1_rotation = optimizable_tensors["rotation"]
        self._level_1_trbf_center = optimizable_tensors["trbf_center"]
        self._level_1_trbf_scale = optimizable_tensors["trbf_scale"]
        # self._level_1_motion = optimizable_tensors["motion"]
        self._level_1_omega = optimizable_tensors["omega"]
        self._level_1_delta_rot_radius = optimizable_tensors["delta_rot_radius"]
        self._level_1_delta_rot_angle_vel = optimizable_tensors["delta_rot_angle_vel"]

        self._level_1_parent_idx = self._level_1_parent_idx[valid_points_mask]

        self.level_1_xyz_gradient_accum = self.level_1_xyz_gradient_accum[valid_points_mask]

        self.level_1_denom = self.level_1_denom[valid_points_mask]
        self.level_1_max_radii2D = self.level_1_max_radii2D[valid_points_mask]
        if self.level_1_omega_mask is not None:
            self.level_1_omega_mask = self.level_1_omega_mask[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"]) == 1 and group["name"] in tensors_dict:
                extension_tensor = tensors_dict[group["name"]]
                stored_state = self.optimizer.state.get(group["params"][0], None)
                if stored_state is not None:

                    stored_state["exp_avg"] = torch.cat(
                        (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                    )
                    stored_state["exp_avg_sq"] = torch.cat(
                        (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0
                    )

                    del self.optimizer.state[group["params"][0]]
                    group["params"][0] = nn.Parameter(
                        torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                    )
                    self.optimizer.state[group["params"][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(
                        torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                    )
                    optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def cat_tensors_to_level_1_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.level_1_optimizer.param_groups:
            if len(group["params"]) == 1 and group["name"] in tensors_dict:
                extension_tensor = tensors_dict[group["name"]]
                stored_state = self.level_1_optimizer.state.get(group["params"][0], None)
                if stored_state is not None:

                    stored_state["exp_avg"] = torch.cat(
                        (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                    )
                    stored_state["exp_avg_sq"] = torch.cat(
                        (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0
                    )

                    del self.level_1_optimizer.state[group["params"][0]]
                    group["params"][0] = nn.Parameter(
                        torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                    )
                    self.level_1_optimizer.state[group["params"][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(
                        torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                    )
                    optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_opacities,
        new_scaling,
        new_rotation,
        new_trbf_center,
        new_trbf_scale,
        new_motion,
        new_omega,
        dummy=None,
    ):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
            "trbf_center": new_trbf_center,
            "trbf_scale": new_trbf_scale,
            "motion": new_motion,
            "omega": new_omega,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        # self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._trbf_center = optimizable_tensors["trbf_center"]
        self._trbf_scale = optimizable_tensors["trbf_scale"]
        self._motion = optimizable_tensors["motion"]
        self._omega = optimizable_tensors["omega"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densification_postfix_level_1(
        self,
        # new_xyz,
        new_parent_idx,
        new_features_dc,
        new_opacities,
        new_scaling,
        new_rotation,
        new_trbf_center,
        new_trbf_scale,
        # new_motion,
        new_omega,
        new_delta_rot_radius,
        new_delta_rot_angle_vel,
        dummy=None,
    ):
        d = {
            # "xyz": new_xyz,
            "f_dc": new_features_dc,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
            "trbf_center": new_trbf_center,
            "trbf_scale": new_trbf_scale,
            # "motion": new_motion,
            "omega": new_omega,
            "delta_rot_radius": new_delta_rot_radius,
            "delta_rot_angle_vel": new_delta_rot_angle_vel,
        }

        optimizable_tensors = self.cat_tensors_to_level_1_optimizer(d)
        self._level_1_features_dc = optimizable_tensors["f_dc"]
        self._level_1_opacity = optimizable_tensors["opacity"]
        self._level_1_scaling = optimizable_tensors["scaling"]
        self._level_1_rotation = optimizable_tensors["rotation"]
        self._level_1_trbf_center = optimizable_tensors["trbf_center"]
        self._level_1_trbf_scale = optimizable_tensors["trbf_scale"]
        # self._level_1_motion = optimizable_tensors["motion"]
        self._level_1_omega = optimizable_tensors["omega"]

        self._level_1_delta_rot_radius = optimizable_tensors["delta_rot_radius"]
        self._level_1_delta_rot_angle_vel = optimizable_tensors["delta_rot_angle_vel"]

        self._level_1_parent_idx = torch.cat((self._level_1_parent_idx, new_parent_idx), dim=0)

        self.level_1_xyz_gradient_accum = torch.zeros((self._level_1_parent_idx.shape[0], 1), device="cuda")
        self.level_1_denom = torch.zeros((self._level_1_parent_idx.shape[0], 1), device="cuda")
        self.level_1_max_radii2D = torch.zeros((self._level_1_parent_idx.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, max_timestamp=1.0, N=2):
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask, torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent
        )
        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        numpy_tmp = rots.cpu().numpy() @ samples.unsqueeze(-1).cpu().numpy()
        # numpy better than cublas..., cublas use stochastic for bmm
        new_xyz = torch.from_numpy(numpy_tmp).cuda().squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)

        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_trbf_center = self._trbf_center[selected_pts_mask].repeat(N, 1)
        new_trbf_center = torch.rand_like(new_trbf_center)
        new_trbf_center = new_trbf_center * max_timestamp
        new_trbf_scale = self._trbf_scale[selected_pts_mask].repeat(N, 1)
        new_motion = self._motion[selected_pts_mask].repeat(N, 1)
        new_omega = self._omega[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_opacity,
            new_scaling,
            new_rotation,
            new_trbf_center,
            new_trbf_scale,
            new_motion,
            new_omega,
        )

        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool))
        )
        self.prune_points(prune_filter)

    def densify_and_split_level_1(self, grads, grad_threshold, scene_extent, max_timestamp=1.0, N=2):
        n_init_points = self.get_level_1_parent_idx.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask, torch.max(self.get_level_1_scaling, dim=1).values > self.percent_dense * scene_extent
        )
        # stds = self.get_level_1_scaling[selected_pts_mask].repeat(N, 1)
        # means = torch.zeros((stds.size(0), 3), device="cuda")
        # samples = torch.normal(mean=means, std=stds)
        # rots = build_rotation(self._level_1_rotation[selected_pts_mask]).repeat(N, 1, 1)
        # numpy_tmp = rots.cpu().numpy() @ samples.unsqueeze(-1).cpu().numpy()
        # # numpy better than cublas..., cublas use stochastic for bmm
        # new_xyz = torch.from_numpy(numpy_tmp).cuda().squeeze(-1) + self.get_level_1_xyz[selected_pts_mask].repeat(N, 1)
        new_parent_idx = self._level_1_parent_idx[selected_pts_mask].repeat(N, 1)

        new_scaling = self.scaling_inverse_activation(
            self.get_level_1_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._level_1_rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._level_1_features_dc[selected_pts_mask].repeat(N, 1)
        new_opacity = self._level_1_opacity[selected_pts_mask].repeat(N, 1)
        new_trbf_center = self._level_1_trbf_center[selected_pts_mask].repeat(N, 1)
        new_trbf_center = torch.rand_like(new_trbf_center)
        new_trbf_center = new_trbf_center * max_timestamp
        new_trbf_scale = self._level_1_trbf_scale[selected_pts_mask].repeat(N, 1)
        # new_motion = self._level_1_motion[selected_pts_mask].repeat(N, 1)
        new_omega = self._level_1_omega[selected_pts_mask].repeat(N, 1)
        new_delta_rot_radius = self._level_1_delta_rot_radius[selected_pts_mask].repeat(N, 1)
        new_delta_rot_angle_vel = self._level_1_delta_rot_angle_vel[selected_pts_mask].repeat(N, 1)

        self.densification_postfix_level_1(
            new_parent_idx,
            new_features_dc,
            new_opacity,
            new_scaling,
            new_rotation,
            new_trbf_center,
            new_trbf_scale,
            # new_motion,
            new_omega,
            new_delta_rot_radius,
            new_delta_rot_angle_vel,
        )
        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool))
        )
        self.prune_points_level_1(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent, max_timestamp=1.0):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)

        selected_pts_mask = torch.logical_and(
            selected_pts_mask, torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent
        )

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_trbf_center = torch.rand((self._trbf_center[selected_pts_mask].shape[0], 1), device="cuda")
        self._trbf_center[selected_pts_mask]
        new_trbf_center = new_trbf_center * max_timestamp
        new_trbf_scale = self._trbf_scale[selected_pts_mask]
        new_motion = self._motion[selected_pts_mask]
        new_omega = self._omega[selected_pts_mask]
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_opacities,
            new_scaling,
            new_rotation,
            new_trbf_center,
            new_trbf_scale,
            new_motion,
            new_omega,
        )

    def densify_and_clone_level_1(self, grads, grad_threshold, scene_extent, max_timestamp=1.0):
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask, torch.max(self.get_level_1_scaling, dim=1).values <= self.percent_dense * scene_extent
        )
        new_parent_idx = self._level_1_parent_idx[selected_pts_mask]
        new_features_dc = self._level_1_features_dc[selected_pts_mask]
        new_opacities = self._level_1_opacity[selected_pts_mask]
        new_scaling = self._level_1_scaling[selected_pts_mask]
        new_rotation = self._level_1_rotation[selected_pts_mask]
        new_trbf_center = torch.rand((self._level_1_trbf_center[selected_pts_mask].shape[0], 1), device="cuda")
        self._level_1_trbf_center[selected_pts_mask]
        new_trbf_center = new_trbf_center * max_timestamp
        new_trbf_scale = self._level_1_trbf_scale[selected_pts_mask]
        # new_motion = self._level_1_motion[selected_pts_mask]
        new_omega = self._level_1_omega[selected_pts_mask]
        new_delta_rot_radius = self._level_1_delta_rot_radius[selected_pts_mask]
        new_delta_rot_angle_vel = self._level_1_delta_rot_angle_vel[selected_pts_mask]
        self.densification_postfix_level_1(
            new_parent_idx,
            new_features_dc,
            new_opacities,
            new_scaling,
            new_rotation,
            new_trbf_center,
            new_trbf_scale,
            # new_motion,
            new_omega,
            new_delta_rot_radius,
            new_delta_rot_angle_vel,
        )

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1

    def add_densification_stats_level_1(self, level_1_viewspace_point_tensor, level_1_update_filter):
        # since in render, all points are used
        # when using transparent level_0, only the level_1 points are used in renderer
        level_1_viewspace_point_tensor_grad = level_1_viewspace_point_tensor.grad
        self.level_1_xyz_gradient_accum[level_1_update_filter] += torch.norm(
            level_1_viewspace_point_tensor_grad[level_1_update_filter, :2], dim=-1, keepdim=True
        )
        self.level_1_denom[level_1_update_filter] += 1

    def densify_and_prune(
        self,
        max_grad,
        min_opacity,
        extent,
        max_screen_size,
        max_timestamp=1.0,
        clone=True,
        split=True,
        prune=True,
        **kwargs,
    ):
        ## raw method from 3dgs debugging hyfluid
        if clone or split:
            grads = self.xyz_gradient_accum / self.denom
            grads[grads.isnan()] = 0.0

        if clone:
            self.densify_and_clone(grads, max_grad, extent, max_timestamp)
        if split:
            self.densify_and_split(grads, max_grad, extent, max_timestamp)

        if prune:
            prune_mask = (self.get_opacity < min_opacity).squeeze()
            if max_screen_size:
                big_points_vs = self.max_radii2D > max_screen_size
                big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
                prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
            self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def post_prune(self, min_opacity, extent, max_screen_size):
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def densify_and_prune_level_1(
        self,
        max_grad,
        min_opacity,
        extent,
        max_screen_size,
        max_timestamp=1.0,
        clone=True,
        split=True,
        prune=True,
        **kwargs,
    ):
        if clone or split:
            grads = self.level_1_xyz_gradient_accum / self.level_1_denom
            grads[grads.isnan()] = 0.0

        if clone:
            self.densify_and_clone_level_1(grads, max_grad, extent, max_timestamp)
        if split:
            self.densify_and_split_level_1(grads, max_grad, extent, max_timestamp)

        if prune:
            prune_mask = (self.get_level_1_opacity < min_opacity).squeeze()
            if max_screen_size:
                big_points_vs = self.level_1_max_radii2D > max_screen_size
                big_points_ws = self.get_level_1_scaling.max(dim=1).values > 0.1 * extent
                prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
            self.prune_points_level_1(prune_mask)

        torch.cuda.empty_cache()

    def post_prune_level_1(self, min_opacity, extent, max_screen_size, prune_min_color=None, prune_max_color=None):
        prune_mask = (self.get_level_1_opacity < min_opacity).squeeze()
        if prune_min_color is not None and isinstance(prune_min_color, float):
            prune_mask_color = (self.get_level_1_features < prune_min_color).squeeze()
            prune_mask = torch.logical_or(prune_mask, prune_mask_color)

        if prune_max_color is not None and isinstance(prune_max_color, float):
            prune_mask_color = (self.get_level_1_features > prune_max_color).squeeze()
            prune_mask = torch.logical_or(prune_mask, prune_mask_color)

        if max_screen_size:
            big_points_vs = self.level_1_max_radii2D > max_screen_size
            big_points_ws = self.get_level_1_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points_level_1(prune_mask)

        torch.cuda.empty_cache()
