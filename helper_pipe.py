def get_render_pipe(option="train_full"):
    print("Render option:", option)
    if option == "train_full":
        from diff_gaussian_rasterization_ch9 import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )

        from gaussian_splatting.renderer import train_full

        return train_full, GaussianRasterizationSettings, GaussianRasterizer

    elif option == "test_full":
        from diff_gaussian_rasterization_ch9 import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )

        from gaussian_splatting.renderer import test_full

        return test_full, GaussianRasterizationSettings, GaussianRasterizer

    elif option == "train_lite":
        from diff_gaussian_rasterization_ch3 import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )

        from gaussian_splatting.renderer import train_lite

        return train_lite, GaussianRasterizationSettings, GaussianRasterizer

    elif option == "test_lite":
        from forward_lite import GaussianRasterizationSettings, GaussianRasterizer

        from gaussian_splatting.renderer import test_lite

        return test_lite, GaussianRasterizationSettings, GaussianRasterizer

    elif option == "test_lite_vis":
        from forward_lite import GaussianRasterizationSettings, GaussianRasterizer

        from gaussian_splatting.renderer import test_lite_vis

        return test_lite_vis, GaussianRasterizationSettings, GaussianRasterizer

    elif option == "train_lite_single":
        from diff_gaussian_rasterization_ch1 import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )

        from gaussian_splatting.renderer import train_lite

        return train_lite, GaussianRasterizationSettings, GaussianRasterizer

    elif option == "test_lite_single_vis":
        from forward_lite_single import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )

        from gaussian_splatting.renderer import test_lite_vis

        return test_lite_vis, GaussianRasterizationSettings, GaussianRasterizer

    elif option == "train_lite_two_level_single":
        from diff_gaussian_rasterization_ch1 import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )

        from gaussian_splatting.renderer import train_lite_two_level

        return train_lite_two_level, GaussianRasterizationSettings, GaussianRasterizer

    elif option == "test_lite_two_level_single_vis":
        from forward_lite_single import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )

        from gaussian_splatting.renderer import test_lite_two_level_vis

        return test_lite_two_level_vis, GaussianRasterizationSettings, GaussianRasterizer

    elif option == "train_lite_act_single":
        from diff_gaussian_rasterization_ch1 import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )

        from gaussian_splatting.renderer import train_lite_act

        return train_lite_act, GaussianRasterizationSettings, GaussianRasterizer

    elif option == "test_lite_act_single_vis":
        from forward_lite_single import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )

        from gaussian_splatting.renderer import test_lite_act_vis

        return test_lite_act_vis, GaussianRasterizationSettings, GaussianRasterizer

    elif option == "train_lite_act_two_sp_level_single":
        from diff_gaussian_rasterization_ch1 import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )

        from gaussian_splatting.renderer import train_lite_act_two_sp_level

        return train_lite_act_two_sp_level, GaussianRasterizationSettings, GaussianRasterizer

    elif option == "test_lite_act_two_sp_level_single_vis":
        from forward_lite_single import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )

        from gaussian_splatting.renderer import test_lite_act_two_sp_level_vis

        return test_lite_act_two_sp_level_vis, GaussianRasterizationSettings, GaussianRasterizer

    elif option == "train_lite_act_two_sp_level_couple_single":
        from diff_gaussian_rasterization_ch1 import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )

        from gaussian_splatting.renderer import train_lite_act_two_sp_level_couple

        return train_lite_act_two_sp_level_couple, GaussianRasterizationSettings, GaussianRasterizer

    elif option == "test_lite_act_two_sp_level_couple_single_vis":
        from forward_lite_single import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )

        from gaussian_splatting.renderer import test_lite_act_two_sp_level_couple_vis

        return test_lite_act_two_sp_level_couple_vis, GaussianRasterizationSettings, GaussianRasterizer

    elif option == "train_lite_act_two_sp_level_couple_transp_single":
        from diff_gaussian_rasterization_ch1 import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )

        from gaussian_splatting.renderer import train_lite_act_two_sp_level_couple_transp

        return train_lite_act_two_sp_level_couple_transp, GaussianRasterizationSettings, GaussianRasterizer

    elif option == "test_lite_act_two_sp_level_couple_transp_single_vis":
        from forward_lite_single import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )

        from gaussian_splatting.renderer import test_lite_act_two_sp_level_couple_transp_vis

        return test_lite_act_two_sp_level_couple_transp_vis, GaussianRasterizationSettings, GaussianRasterizer

    elif option == "train_lite_act_two_sp_level_couple_transp_zerodel_single":
        from diff_gaussian_rasterization_ch1 import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )

        from gaussian_splatting.renderer import train_lite_act_two_sp_level_couple_transp_zerodel

        return train_lite_act_two_sp_level_couple_transp_zerodel, GaussianRasterizationSettings, GaussianRasterizer

    elif option == "test_lite_act_two_sp_level_couple_transp_zerodel_single_vis":
        from forward_lite_single import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )

        from gaussian_splatting.renderer import test_lite_act_two_sp_level_couple_transp_zerodel_vis

        return test_lite_act_two_sp_level_couple_transp_zerodel_vis, GaussianRasterizationSettings, GaussianRasterizer

    elif option == "train_lite_act_two_sp_level_couple_transp_zerodel_trbfs_single":
        from diff_gaussian_rasterization_ch1 import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )

        from gaussian_splatting.renderer import train_lite_act_two_sp_level_couple_transp_zerodel_trbfs

        return train_lite_act_two_sp_level_couple_transp_zerodel_trbfs, GaussianRasterizationSettings, GaussianRasterizer

    elif option == "test_lite_act_two_sp_level_couple_transp_zerodel_trbfs_single_vis":
        from forward_lite_single import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )

        from gaussian_splatting.renderer import test_lite_act_two_sp_level_couple_transp_zerodel_trbfs_vis

        return test_lite_act_two_sp_level_couple_transp_zerodel_trbfs_vis, GaussianRasterizationSettings, GaussianRasterizer

    elif option == "train_lite_act_two_sp_level_couple_transp_rotdel_single":
        from diff_gaussian_rasterization_ch1 import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )

        from gaussian_splatting.renderer import train_lite_act_two_sp_level_couple_transp_rotdel

        return train_lite_act_two_sp_level_couple_transp_rotdel, GaussianRasterizationSettings, GaussianRasterizer

    elif option == "test_lite_act_two_sp_level_couple_transp_rotdel_single_vis":
        from forward_lite_noprepre_single import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )

        from gaussian_splatting.renderer import test_lite_act_two_sp_level_couple_transp_rotdel_vis

        return test_lite_act_two_sp_level_couple_transp_rotdel_vis, GaussianRasterizationSettings, GaussianRasterizer

    elif option == "train_lite_act_two_sp_level_couple_transp_rotdel_trbfs_single":
        from diff_gaussian_rasterization_ch1 import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )

        from gaussian_splatting.renderer import train_lite_act_two_sp_level_couple_transp_rotdel_trbfs

        return train_lite_act_two_sp_level_couple_transp_rotdel_trbfs, GaussianRasterizationSettings, GaussianRasterizer

    elif option == "test_lite_act_two_sp_level_couple_transp_rotdel_trbfs_single_vis":
        from forward_lite_noprepre_single import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )

        from gaussian_splatting.renderer import test_lite_act_two_sp_level_couple_transp_rotdel_trbfs_vis

        return test_lite_act_two_sp_level_couple_transp_rotdel_trbfs_vis, GaussianRasterizationSettings, GaussianRasterizer

    elif option == "train_lite_act_two_sp_level_couple_transp_trotdel_trbfs_single":
        from diff_gaussian_rasterization_ch1 import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )

        from gaussian_splatting.renderer import train_lite_act_two_sp_level_couple_transp_trotdel_trbfs

        return train_lite_act_two_sp_level_couple_transp_trotdel_trbfs, GaussianRasterizationSettings, GaussianRasterizer

    elif option == "test_lite_act_two_sp_level_couple_transp_trotdel_trbfs_single_vis":
        from forward_lite_noprepre_single import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )

        from gaussian_splatting.renderer import test_lite_act_two_sp_level_couple_transp_trotdel_trbfs_vis

        return test_lite_act_two_sp_level_couple_transp_trotdel_trbfs_vis, GaussianRasterizationSettings, GaussianRasterizer

    elif option == "train_lite_single_all":
        from diff_gaussian_rasterization_ch1 import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )

        from gaussian_splatting.renderer import train_lite_all

        return train_lite_all, GaussianRasterizationSettings, GaussianRasterizer

    elif option == "test_lite_single_all_vis":
        from forward_lite_single_all import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )

        from gaussian_splatting.renderer import test_lite_all_vis

        return test_lite_all_vis, GaussianRasterizationSettings, GaussianRasterizer

    elif option == "train_lite_single_trbf_center":
        from diff_gaussian_rasterization_ch1 import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )

        from gaussian_splatting.renderer import train_lite_trbf_center

        return train_lite_trbf_center, GaussianRasterizationSettings, GaussianRasterizer

    elif option == "test_lite_single_trbf_center_vis":
        from forward_lite_single_trbf_center import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )

        from gaussian_splatting.renderer import test_lite_trbf_center_vis

        return test_lite_trbf_center_vis, GaussianRasterizationSettings, GaussianRasterizer

    elif option == "train_lite_single_xyz_quadric":
        from diff_gaussian_rasterization_ch1 import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )

        from gaussian_splatting.renderer import train_lite_xyz_quadric

        return train_lite_xyz_quadric, GaussianRasterizationSettings, GaussianRasterizer

    elif option == "test_lite_single_xyz_quadric_vis":
        from forward_lite_single_xyz_quadric import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )

        from gaussian_splatting.renderer import test_lite_xyz_quadric_vis

        return test_lite_xyz_quadric_vis, GaussianRasterizationSettings, GaussianRasterizer

    elif option == "train_lite_single_xyz_quadric_trbf_center":
        from diff_gaussian_rasterization_ch1 import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )

        from gaussian_splatting.renderer import train_lite_xyz_quadric_trbf_center

        return train_lite_xyz_quadric_trbf_center, GaussianRasterizationSettings, GaussianRasterizer

    elif option == "test_lite_single_xyz_quadric_trbf_center_vis":
        from forward_lite_single_xyz_quadric_trbf_center import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )

        from gaussian_splatting.renderer import test_lite_xyz_quadric_trbf_center_vis

        return test_lite_xyz_quadric_trbf_center_vis, GaussianRasterizationSettings, GaussianRasterizer

    elif option == "train_lite_single_xyz_linear":
        from diff_gaussian_rasterization_ch1 import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )

        from gaussian_splatting.renderer import train_lite_xyz_linear

        return train_lite_xyz_linear, GaussianRasterizationSettings, GaussianRasterizer

    elif option == "test_lite_single_xyz_linear_vis":
        from forward_lite_single_xyz_linear import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )

        from gaussian_splatting.renderer import test_lite_xyz_linear_vis

        return test_lite_xyz_linear_vis, GaussianRasterizationSettings, GaussianRasterizer

    elif option == "train_lite_single_xyz_linear_color":
        from diff_gaussian_rasterization_ch1 import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )

        from gaussian_splatting.renderer import train_lite_xyz_linear_color

        return train_lite_xyz_linear_color, GaussianRasterizationSettings, GaussianRasterizer

    elif option == "test_lite_single_xyz_linear_color_vis":
        from forward_lite_single_xyz_linear_color import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )

        from gaussian_splatting.renderer import test_lite_xyz_linear_color_vis

        return test_lite_xyz_linear_color_vis, GaussianRasterizationSettings, GaussianRasterizer

    elif option == "train_lite_single_xyz_linear_color_trbf_c_act":
        from diff_gaussian_rasterization_ch1 import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )

        from gaussian_splatting.renderer import train_lite_xyz_linear_color_trbf_c_act

        return train_lite_xyz_linear_color_trbf_c_act, GaussianRasterizationSettings, GaussianRasterizer

    elif option == "test_lite_single_xyz_linear_color_trbf_c_act_vis":
        from forward_lite_single_xyz_linear_color_trbf_c_act import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )

        from gaussian_splatting.renderer import (
            test_lite_xyz_linear_color_trbf_c_act_vis,
        )

        return test_lite_xyz_linear_color_trbf_c_act_vis, GaussianRasterizationSettings, GaussianRasterizer

    elif option == "train_lite_single_xyz_linear_color_trbf_c_act_xyz":
        from diff_gaussian_rasterization_ch1 import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )

        from gaussian_splatting.renderer import (
            train_lite_xyz_linear_color_trbf_c_act_xyz,
        )

        return train_lite_xyz_linear_color_trbf_c_act_xyz, GaussianRasterizationSettings, GaussianRasterizer

    elif option == "test_lite_single_xyz_linear_color_trbf_c_act_xyz_vis":
        from forward_lite_single_xyz_linear_color_trbf_c_act import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )

        from gaussian_splatting.renderer import (
            test_lite_xyz_linear_color_trbf_c_act_xyz_vis,
        )

        return test_lite_xyz_linear_color_trbf_c_act_xyz_vis, GaussianRasterizationSettings, GaussianRasterizer

    elif option == "train_lite_single_xyz_linear_color_trbf_center":
        from diff_gaussian_rasterization_ch1 import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )

        from gaussian_splatting.renderer import train_lite_xyz_linear_color_trbf_center

        return train_lite_xyz_linear_color_trbf_center, GaussianRasterizationSettings, GaussianRasterizer

    elif option == "test_lite_single_xyz_linear_color_trbf_center_vis":
        from forward_lite_single_xyz_linear_color_trbf_center import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )

        from gaussian_splatting.renderer import (
            test_lite_xyz_linear_color_trbf_center_vis,
        )

        return test_lite_xyz_linear_color_trbf_center_vis, GaussianRasterizationSettings, GaussianRasterizer

    elif option == "train_lite_single_xyz_linear_color_source":
        from diff_gaussian_rasterization_ch1 import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )

        from gaussian_splatting.renderer import train_lite_xyz_linear_color_source

        return train_lite_xyz_linear_color_source, GaussianRasterizationSettings, GaussianRasterizer

    elif option == "test_lite_single_xyz_linear_color_source_vis":
        from forward_lite_single_xyz_linear_color_source import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )

        from gaussian_splatting.renderer import test_lite_xyz_linear_color_source_vis

        return test_lite_xyz_linear_color_source_vis, GaussianRasterizationSettings, GaussianRasterizer

    elif option == "train_lite_single_opacity_no_t":
        from diff_gaussian_rasterization_ch1 import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )

        from gaussian_splatting.renderer import train_lite_opacity_no_t

        return train_lite_opacity_no_t, GaussianRasterizationSettings, GaussianRasterizer

    elif option == "test_lite_single_opacity_no_t_vis":
        from forward_lite_single_opacity_no_t import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )

        from gaussian_splatting.renderer import test_lite_opacity_no_t_vis

        return test_lite_opacity_no_t_vis, GaussianRasterizationSettings, GaussianRasterizer

    elif option == "train_lite_single_opacity_exp_linear":
        from diff_gaussian_rasterization_ch1 import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )

        from gaussian_splatting.renderer import train_lite_opacity_exp_linear

        return train_lite_opacity_exp_linear, GaussianRasterizationSettings, GaussianRasterizer

    elif option == "test_lite_single_opacity_exp_linear_vis":
        from forward_lite_single_opacity_exp_linear import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )

        from gaussian_splatting.renderer import test_lite_opacity_exp_linear_vis

        return test_lite_opacity_exp_linear_vis, GaussianRasterizationSettings, GaussianRasterizer

    elif option == "train_lite_single_opacity_linear":
        from diff_gaussian_rasterization_ch1 import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )

        from gaussian_splatting.renderer import train_lite_opacity_linear

        return train_lite_opacity_linear, GaussianRasterizationSettings, GaussianRasterizer

    elif option == "test_lite_single_opacity_linear_vis":
        from forward_lite_single_opacity_linear import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )

        from gaussian_splatting.renderer import test_lite_opacity_linear_vis

        return test_lite_opacity_linear_vis, GaussianRasterizationSettings, GaussianRasterizer

    elif option == "test_lite_single_vis":
        from forward_lite_single import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )

        from gaussian_splatting.renderer import test_lite_vis

        return test_lite_vis, GaussianRasterizationSettings, GaussianRasterizer

    else:
        raise NotImplementedError("Render {} not implemented".format(option))
