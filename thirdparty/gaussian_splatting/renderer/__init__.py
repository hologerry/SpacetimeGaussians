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

#######################################################################################################################
##### NOTE: CODE IN THIS FILE IS NOT INCLUDED IN THE OVERALL PROJECT'S MIT LICENSE ####################################
##### USE OF THIS CODE FOLLOWS THE COPYRIGHT NOTICE ABOVE #####
#######################################################################################################################


from .ours_full import test_ours_full, train_ours_full
from .ours_lite import test_ours_lite_vis, train_ours_lite
from .simple_color_scale_rotation_act import test_ours_lite_act_vis, train_ours_lite_act
from .simple_opacity_exp_linear import (
    test_ours_lite_opacity_exp_linear_vis,
    train_ours_lite_opacity_exp_linear,
)
from .simple_opacity_linear import (
    test_ours_lite_opacity_linear_vis,
    train_ours_lite_opacity_linear,
)
from .simple_opacity_no_t import (
    test_ours_lite_opacity_no_t_vis,
    train_ours_lite_opacity_no_t,
)
from .simple_trbf_center import (
    test_ours_lite_trbf_center_vis,
    train_ours_lite_trbf_center,
)
from .simple_xyz_linear import test_ours_lite_xyz_linear_vis, train_ours_lite_xyz_linear
from .simple_xyz_linear_color import (
    test_ours_lite_xyz_linear_color_vis,
    train_ours_lite_xyz_linear_color,
)
from .simple_xyz_linear_color_source import (
    test_ours_lite_xyz_linear_color_source_vis,
    train_ours_lite_xyz_linear_color_source,
)
from .simple_xyz_linear_color_trbf_c_act import (
    test_ours_lite_xyz_linear_color_trbf_c_act_vis,
    train_ours_lite_xyz_linear_color_trbf_c_act,
)
from .simple_xyz_linear_color_trbf_c_act_xyz import (
    test_ours_lite_xyz_linear_color_trbf_c_act_xyz_vis,
    train_ours_lite_xyz_linear_color_trbf_c_act_xyz,
)
from .simple_xyz_linear_color_trbf_center import (
    test_ours_lite_xyz_linear_color_trbf_center_vis,
    train_ours_lite_xyz_linear_color_trbf_center,
)
from .simple_xyz_quadric import (
    test_ours_lite_xyz_quadric_vis,
    train_ours_lite_xyz_quadric,
)
from .simple_xyz_quadric_trbf_center import (
    test_ours_lite_xyz_quadric_trbf_center_vis,
    train_ours_lite_xyz_quadric_trbf_center,
)
