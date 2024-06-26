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

from argparse import ArgumentParser


class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group


class ModelParams(ParamGroup):

    def export_changed_args_to_json(self, args):
        defaults = {}
        for arg in vars(args).items():
            try:
                if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                    default_value = getattr(self, arg[0])
                    # defaults[ arg[0] ] = default_value
                    if default_value != arg[1]:
                        defaults[arg[0]] = arg[1]
            except:
                pass

        return defaults

    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.verify_llff = 0
        self.eval = False
        self.model = "g_model"
        self.loader = "colmap"

        self.basic_function = ""
        self.densify = 0

        self.rgb_function = "none"

        self.start_time = 0
        self.duration = 50
        self.time_step = 1
        self.max_timestamp = 1.0

        # GaussianFluid parameters
        self.grey_image = False
        # four training cameras, 00, 01, 03, 04
        self.train_views = "0134"
        # same format as train views # the views that produced from zero123 finetuned model
        self.train_views_fake = None
        self.use_best_fake = False
        self.source_init = False
        self.new_pts = 10_000
        self.img_offset = False
        self.init_region_type = "large"

        self.init_num_pts_per_time = 1000
        self.init_trbf_c_fix = False
        self.init_color_fix_value = None  # None for random color, float for fix value

        self.level_1_init_num_pts_per_time = 100
        self.level_1_init_pts_op = 0.1
        self.level_1_init_pts_color = 0.5
        self.level_1_init_pts_xyz = "parent"
        self.level_1_init_pts_xyz_offset = 1e-3  # random offset on parent xyz
        self.level_1_init_pts_scale = "dist"  # or float, such as -5, before exp
        self.level_1_init_pts_min_opacity = 0.05
        self.level_1_init_pts_delta_rot_radius_scale = 5.0  # radius ratio scale on mean scales
        self.level_1_init_pts_delta_rot_angle_vel_rand = None
        self.level_1_init_pts_fix_trbfs = 2.0
        self.level_1_delta_rot_type = "xz"
        self.level_1_delta_sin_a = 1.0
        self.level_1_delta_sin_omega = 1.0
        self.level_1_delta_sin_phi = 0.0
        self.level_1_init_num_pts_per_parent = 10
        self.level_1_init_pts_delta_rig_sur_radius_scale = 3.0
        self.level_1_init_delta_x_max = 0.005
        self.level_1_init_delta_y_max = 0.005
        self.level_1_init_delta_z_max = 0.005

        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.rd_pipe = "v2"
        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000

        self.batch = 2

        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.feature_t_lr = 0.001
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005

        self.trbf_c_lr = 0.0001
        self.trbf_s_lr = 0.03
        self.trbf_scale_init = 0.0
        self.rgb_lr = 0.0001

        self.move_lr = 3.5

        self.omega_lr = 0.0001
        self.beta_lr = 0.0001
        self.rotation_lr = 0.001

        self.level_1_position_lr_init = 0.00016
        self.level_1_position_lr_final = 0.0000016
        self.level_1_position_lr_delay_mult = 0.01
        self.level_1_position_lr_max_steps = 30_000
        self.level_1_feature_lr = 0.0025
        self.level_1_feature_t_lr = 0.001
        self.level_1_opacity_lr = 0.05
        self.level_1_scaling_lr = 0.005

        self.level_1_color_a1_lr = 0.0025
        self.level_1_color_c1_lr = 0.0025
        self.level_1_color_a2_lr = 0.0025
        self.level_1_color_c2_lr = 0.0025
        self.level_1_color_a3_lr = 0.0025
        self.level_1_color_c3_lr = 0.0025

        self.level_1_trbf_c_lr = 0.0001
        self.level_1_trbf_s_lr = 0.03
        self.level_1_trbf_scale_init = 0.0
        self.level_1_rgb_lr = 0.0001

        self.level_1_move_lr = 3.5

        self.level_1_omega_lr = 0.0001
        self.level_1_beta_lr = 0.0001
        self.level_1_rotation_lr = 0.001

        self.level_1_delta_rot_radius_lr = 0.0025
        self.level_1_delta_rot_angle_vel_lr = 0.002

        self.level_1_delta_trot_center_lr = 0.0025
        self.level_1_delta_trot_radius_lr = 0.0025
        self.level_1_delta_trot_alpha_lr = 0.0025
        self.level_1_delta_trot_angle_vel_lr = 0.003
        self.level_1_delta_trot_beta_lr = 0.003

        self.level_1_delta_sin_a_lr = 0.0025
        self.level_1_delta_sin_omega_lr = 0.0025
        self.level_1_delta_sin_phi_lr = 0.0025

        self.level_1_delta_rig_sur_radius_lr = 0.0025
        self.level_1_delta_rig_sur_azimuth_lr = 0.0025
        self.level_1_delta_rig_sur_polar_lr = 0.0025

        self.level_1_delta_xyz_lr_init = 0.00016
        self.level_1_delta_xyz_lr_final = 0.0000016
        self.level_1_delta_xyz_lr_delay_mult = 0.01
        self.level_1_delta_xyz_lr_max_steps = 30_000

        self.lambda_dssim = 0.2

        self.percent_dense = 0.01

        self.opacity_reset_interval = 3_000
        self.opacity_reset_at = 10000

        self.densify_cnt = 6
        self.reg = 0
        self.lambda_reg = 0.0001
        self.shrink_scale = 2.0
        self.random_feature = 0
        self.ems_type = 0
        self.radials = 10.0
        self.new_ray_step = 2
        self.ems_start = 1600  # small for debug
        self.loss_tart = 200
        self.save_emp_points = 0
        self.prune_by_size = 0
        self.ems_threshold = 0.6
        self.opacity_threshold = 0.005
        self.selective_view = 0
        self.preprocess_points = 0
        self.freeze_rotation_iteration = 8001
        self.add_sph_points_scale = 0.8
        self.g_num_limit = 330000
        self.ray_end = 7.5
        self.ray_start = 0.7
        self.shuffle_ems = 1
        self.prev_path = "1"
        self.load_all = 0
        self.remove_scale = 5
        self.gt_mask = 0  # 0 means not train with mask for undistorted gt image; 1 means

        # hyfluid
        self.cur_time_only_iterations = 10000
        self.iterations_per_time = 250
        self.iterations_per_time_post = 12

        self.lambda_velocity = 0.0
        self.lambda_opacity_vel = 0.0

        self.densification_interval = 100
        self.densify_from_iter = 500
        self.densify_until_iter = 9000
        self.densify_grad_threshold = 0.0002

        self.clone = True
        self.split = True
        self.split_prune = True
        self.prune = True

        self.post_prune = False
        self.post_prune_interval = 100
        self.post_prune_from_iter = 25000
        self.post_prune_until_iter = 27000

        self.zero_grad_level = None

        self.level_1_start_iter = 30000
        self.level_1_clone = True
        self.level_1_split = True
        self.level_1_split_prune = True
        self.level_1_prune = True
        self.level_1_densify_from_iter = 30000
        self.level_1_densify_until_iter = 35000

        self.level_1_post_prune = False
        self.level_1_post_prune_min_color = None
        self.level_1_post_prune_max_color = None
        self.level_1_post_prune_interval = 100
        self.level_1_post_prune_from_iter = 55000
        self.level_1_post_prune_until_iter = 57000

        self.act_level_1 = False

        self.transparent_level_0 = False

        self.lambda_level_1_motion = 0
        self.lambda_level_1_delta_xyz = 0

        # self.two_level_joint_start_iter = 60000
        self.lambda_level_1_delta_xyz_smooth = 0

        self.lambda_level_1_color_smooth = 0
        self.lambda_level_1_scale_reg = 0
        self.lambda_level_1_scale_reg_ratio = 0

        super().__init__(parser, "Optimization Parameters")
