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


from .pipe_full import test_full, train_full
from .pipe_lite import test_lite, test_lite_vis, train_lite
from .pipe_lite_act import test_lite_act_vis, train_lite_act
from .pipe_lite_act_two_sp_level import (
    test_lite_act_two_sp_level_vis,
    train_lite_act_two_sp_level,
)
from .pipe_lite_act_two_sp_level_couple import (
    test_lite_act_two_sp_level_couple_vis,
    train_lite_act_two_sp_level_couple,
)
from .pipe_lite_act_two_sp_level_couple_transp import (
    test_lite_act_two_sp_level_couple_transp_vis,
    train_lite_act_two_sp_level_couple_transp,
)
from .pipe_lite_act_two_sp_level_couple_transp_trbfs import (
    test_lite_act_two_sp_level_couple_transp_trbfs_vis,
    train_lite_act_two_sp_level_couple_transp_trbfs,
)
from .pipe_lite_act_two_sp_level_couple_transp_zerodel import (
    sim_lite_act_two_sp_level_couple_transp_zerodel_vis,
    test_lite_act_two_sp_level_couple_transp_zerodel_vis,
    train_lite_act_two_sp_level_couple_transp_zerodel,
)
from .pipe_lite_act_two_sp_level_couple_transp_zerodel_trbfs import (
    test_lite_act_two_sp_level_couple_transp_zerodel_trbfs_vis,
    train_lite_act_two_sp_level_couple_transp_zerodel_trbfs,
)
from .pipe_lite_act_two_sp_level_couple_transp_zerodel_topacity import (
    test_lite_act_two_sp_level_couple_transp_zerodel_topacity_vis,
    train_lite_act_two_sp_level_couple_transp_zerodel_topacity,
)
from .pipe_lite_act_two_sp_level_couple_transp_zerodel_tcolor import (
    test_lite_act_two_sp_level_couple_transp_zerodel_tcolor_vis,
    train_lite_act_two_sp_level_couple_transp_zerodel_tcolor,
)
from .pipe_lite_act_two_sp_level_couple_transp_zerodel_cucolor import (
    test_lite_act_two_sp_level_couple_transp_zerodel_cucolor_vis,
    train_lite_act_two_sp_level_couple_transp_zerodel_cucolor,
)
from .pipe_lite_act_two_sp_level_couple_transp_rotdel import (
    test_lite_act_two_sp_level_couple_transp_rotdel_vis,
    train_lite_act_two_sp_level_couple_transp_rotdel,
)
from .pipe_lite_act_two_sp_level_couple_transp_rotdel_trbfs import (
    test_lite_act_two_sp_level_couple_transp_rotdel_trbfs_vis,
    train_lite_act_two_sp_level_couple_transp_rotdel_trbfs,
)
from .pipe_lite_act_two_sp_level_couple_transp_trotdel_trbfs import (
    test_lite_act_two_sp_level_couple_transp_trotdel_trbfs_vis,
    train_lite_act_two_sp_level_couple_transp_trotdel_trbfs,
)
from .pipe_lite_act_two_sp_level_couple_transp_sindel_trbfs import (
    test_lite_act_two_sp_level_couple_transp_sindel_trbfs_vis,
    train_lite_act_two_sp_level_couple_transp_sindel_trbfs,
)
from .pipe_lite_act_two_sp_level_couple_transp_linsindel_trbfs import (
    test_lite_act_two_sp_level_couple_transp_linsindel_trbfs_vis,
    train_lite_act_two_sp_level_couple_transp_linsindel_trbfs,
)
from .pipe_lite_act_two_sp_level_couple_transp_rigsurdel import (
    test_lite_act_two_sp_level_couple_transp_rigsurdel_vis,
    train_lite_act_two_sp_level_couple_transp_rigsurdel,
)
from .pipe_lite_act_two_sp_level_couple_transp_rigsurdel_topacity import (
    test_lite_act_two_sp_level_couple_transp_rigsurdel_topacity_vis,
    train_lite_act_two_sp_level_couple_transp_rigsurdel_topacity,
)
from .pipe_lite_act_two_sp_level_couple_transp_rigsurdel_tcolor import (
    test_lite_act_two_sp_level_couple_transp_rigsurdel_tcolor_vis,
    train_lite_act_two_sp_level_couple_transp_rigsurdel_tcolor,
)
from .pipe_lite_act_two_sp_level_couple_transp_rigsurdel_trbfs import (
    test_lite_act_two_sp_level_couple_transp_rigsurdel_trbfs_vis,
    train_lite_act_two_sp_level_couple_transp_rigsurdel_trbfs,
)
from .pipe_lite_act_two_sp_level_couple_transp_tlearndel import (
    test_lite_act_two_sp_level_couple_transp_tlearndel_vis,
    train_lite_act_two_sp_level_couple_transp_tlearndel,
)
from .pipe_lite_act_two_sp_level_couple_transp_tlearndel_trbfs import (
    test_lite_act_two_sp_level_couple_transp_tlearndel_trbfs_vis,
    train_lite_act_two_sp_level_couple_transp_tlearndel_trbfs,
)
from .pipe_lite_two_level import test_lite_two_level_vis, train_lite_two_level
from .pipe_simple_opacity_exp_linear import (
    test_lite_opacity_exp_linear_vis,
    train_lite_opacity_exp_linear,
)
from .pipe_simple_opacity_linear import (
    test_lite_opacity_linear_vis,
    train_lite_opacity_linear,
)
from .pipe_simple_opacity_no_t import (
    test_lite_opacity_no_t_vis,
    train_lite_opacity_no_t,
)
from .pipe_simple_trbf_center import test_lite_trbf_center_vis, train_lite_trbf_center
from .pipe_simple_xyz_linear import test_lite_xyz_linear_vis, train_lite_xyz_linear
from .pipe_simple_xyz_linear_color import (
    test_lite_xyz_linear_color_vis,
    train_lite_xyz_linear_color,
)
from .pipe_simple_xyz_linear_color_source import (
    test_lite_xyz_linear_color_source_vis,
    train_lite_xyz_linear_color_source,
)
from .pipe_simple_xyz_linear_color_trbf_c_act import (
    test_lite_xyz_linear_color_trbf_c_act_vis,
    train_lite_xyz_linear_color_trbf_c_act,
)
from .pipe_simple_xyz_linear_color_trbf_c_act_xyz import (
    test_lite_xyz_linear_color_trbf_c_act_xyz_vis,
    train_lite_xyz_linear_color_trbf_c_act_xyz,
)
from .pipe_simple_xyz_linear_color_trbf_center import (
    test_lite_xyz_linear_color_trbf_center_vis,
    train_lite_xyz_linear_color_trbf_center,
)
from .pipe_simple_xyz_quadric import test_lite_xyz_quadric_vis, train_lite_xyz_quadric
from .pipe_simple_xyz_quadric_trbf_center import (
    test_lite_xyz_quadric_trbf_center_vis,
    train_lite_xyz_quadric_trbf_center,
)
