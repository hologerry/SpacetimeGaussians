import math
import time

import torch

from gaussian_splatting.gaussian.gm_simple_color_scale_rotation_act_two_sp_level_couple_transp_zerodel_cucolor import (
    GaussianModel,
)


def train_lite_act_two_sp_level_couple_transp_zerodel_cucolor(
    viewpoint_camera,
    gm: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    basic_function=None,
    GRsetting=None,
    GRzer=None,
    level=0,
    act_level_1=False,
    transp_level_0=False,
    **kwargs,
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    # when set act_level_1, activation is applied on all points, otherwise only on the level 0 points

    # Set up rasterization configuration
    tan_fov_x = math.tan(viewpoint_camera.FoVx * 0.5)
    tan_fov_y = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GRsetting(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tan_fov_x=tan_fov_x,
        tan_fov_y=tan_fov_y,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        view_matrix=viewpoint_camera.world_view_transform,
        proj_matrix=viewpoint_camera.full_proj_transform,
        sh_degree=gm.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
    )

    rasterizer = GRzer(raster_settings=raster_settings)

    level_0_means3D = gm.get_xyz

    n_level_0_points = level_0_means3D.shape[0]

    level_0_point_opacity = gm.get_opacity
    level_0_trbf_center = gm.get_trbf_center
    level_0_trbf_scale = gm.get_trbf_scale

    level_0_trbf_distance_offset = viewpoint_camera.timestamp - level_0_trbf_center

    level_0_time_coefficient = gm.t_activation(level_0_trbf_distance_offset)
    if not act_level_1:
        level_0_time_coefficient[n_level_0_points:] = 1.0

    level_0_trbf_distance = level_0_trbf_distance_offset / torch.exp(level_0_trbf_scale)
    level_0_trbf_output = basic_function(level_0_trbf_distance)

    level_0_opacity = level_0_point_opacity * level_0_trbf_output * level_0_time_coefficient
    gm.trbf_output = level_0_trbf_output

    level_0_scales = gm.get_scaling
    level_0_scales = level_0_scales * level_0_time_coefficient

    level_0_motion = gm.get_motion

    level_0_tforpoly = level_0_trbf_distance_offset.detach()

    level_0_cur_means3D = (
        level_0_means3D
        + level_0_motion[:, 0:3] * level_0_tforpoly * level_0_time_coefficient
        + level_0_motion[:, 3:6] * level_0_tforpoly * level_0_tforpoly * level_0_time_coefficient
        + level_0_motion[:, 6:9] * level_0_tforpoly * level_0_tforpoly * level_0_tforpoly * level_0_time_coefficient
    )

    level_0_rotations = gm.get_rotation(level_0_tforpoly)
    level_0_rotations = level_0_rotations * level_0_time_coefficient
    level_0_colors_precomp = gm.get_features
    level_0_colors_precomp = level_0_colors_precomp * level_0_time_coefficient

    if level == 0:

        n_points = n_level_0_points
        cur_means3D = level_0_cur_means3D
        rotations = level_0_rotations
        colors_precomp = level_0_colors_precomp
        opacity = level_0_opacity
        scales = level_0_scales

    elif level == 1:

        level_1_parent_idx = gm.get_level_1_parent_idx

        n_level_1_points = level_1_parent_idx.shape[0]
        n_points = n_level_0_points + n_level_1_points

        level_1_point_opacity = gm.get_level_1_opacity
        level_1_trbf_center = gm.get_level_1_trbf_center
        level_1_trbf_scale = gm.get_level_1_trbf_scale

        level_1_trbf_distance_offset = viewpoint_camera.timestamp - level_1_trbf_center
        if act_level_1:
            level_1_time_coefficient = gm.t_activation(level_1_trbf_distance_offset)
        else:
            level_1_time_coefficient = torch.ones_like(level_1_trbf_distance_offset)

        level_1_trbf_distance = level_1_trbf_distance_offset / torch.exp(level_1_trbf_scale)
        level_1_trbf_output = basic_function(level_1_trbf_distance)
        gm.level_1_trbf_output = level_1_trbf_output

        level_1_opacity = level_1_point_opacity * level_1_trbf_output * level_1_time_coefficient

        level_1_scales = gm.get_level_1_scaling
        level_1_scales = level_1_scales * level_1_time_coefficient

        level_1_motion = gm.get_level_1_motion
        level_1_tforpoly = level_1_trbf_distance_offset.detach()

        level_1_parent_idx = level_1_parent_idx.squeeze(1)

        level_1_parent_means3D = level_0_cur_means3D[level_1_parent_idx]

        level_1_delta_means3D = torch.zeros_like(level_1_parent_means3D)

        level_1_cur_means3D = level_1_parent_means3D + level_1_delta_means3D

        level_1_rotations = gm.get_level_1_rotation(level_1_tforpoly)
        level_1_rotations = level_1_rotations * level_1_time_coefficient
        level_1_colors_c = gm.get_level_1_features
        level_1_color_a1 = gm.get_level_1_color_a1
        level_1_color_c1 = gm.get_level_1_color_c1
        level_1_color_a2 = gm.get_level_1_color_a2
        level_1_color_c2 = gm.get_level_1_color_c2
        level_1_color_a3 = gm.get_level_1_color_a3
        level_1_color_c3 = gm.get_level_1_color_c3
        level_1_cubic_color = level_1_colors_c + level_1_color_a1 * (viewpoint_camera.timestamp - level_1_color_c1)
        level_1_cubic_color += level_1_color_a2 * (viewpoint_camera.timestamp - level_1_color_c2) ** 2
        level_1_cubic_color += level_1_color_a3 * (viewpoint_camera.timestamp - level_1_color_c3) ** 3
        level_1_cubic_color = torch.clamp(level_1_cubic_color, -0.5, 1.5)
        level_1_colors_precomp = level_1_cubic_color * level_1_time_coefficient

        if transp_level_0:
            # only the level_1 points are used in the rendering
            cur_means3D = level_1_cur_means3D
            rotations = level_1_rotations
            colors_precomp = level_1_colors_precomp
            opacity = level_1_opacity
            scales = level_1_scales
            n_points = n_level_1_points

        else:
            cur_means3D = torch.cat((level_0_cur_means3D, level_1_cur_means3D), dim=0)
            rotations = torch.cat((level_0_rotations, level_1_rotations), dim=0)
            colors_precomp = torch.cat((level_0_colors_precomp, level_1_colors_precomp), dim=0)
            opacity = torch.cat((level_0_opacity, level_1_opacity), dim=0)
            scales = torch.cat((level_0_scales, level_1_scales), dim=0)

    else:
        raise ValueError(f"Invalid level: {level}")

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screen_space_points = torch.zeros((n_points, 3), dtype=cur_means3D.dtype, requires_grad=True, device="cuda") + 0
    try:
        screen_space_points.retain_grad()
    except:
        pass

    means2D = screen_space_points

    rendered_image, radii, depth = rasterizer(
        means3D=cur_means3D,
        means2D=means2D,
        shs=None,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
    )

    output_dict = {
        "render": rendered_image,
        "viewspace_points": screen_space_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "opacity": opacity,
        "depth": depth,
    }

    if level == 1:
        output_dict["level_1_delta_means3D"] = level_1_delta_means3D

    return output_dict


def test_lite_act_two_sp_level_couple_transp_zerodel_cucolor_vis(
    viewpoint_camera,
    gm: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    basic_function=None,
    GRsetting=None,
    GRzer=None,
    level=0,
    act_level_1=False,
    transp_level_0=False,
    **kwargs,
):

    torch.cuda.synchronize()
    start_time = time.time()

    tan_fov_x = math.tan(viewpoint_camera.FoVx * 0.5)
    tan_fov_y = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GRsetting(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tan_fov_x=tan_fov_x,
        tan_fov_y=tan_fov_y,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        view_matrix=viewpoint_camera.world_view_transform,
        proj_matrix=viewpoint_camera.full_proj_transform,
        sh_degree=gm.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
    )

    rasterizer = GRzer(raster_settings=raster_settings)

    level_0_means3D = gm.get_xyz

    n_level_0_points = level_0_means3D.shape[0]

    level_0_trbf_center = gm.get_trbf_center
    level_0_tforpoly = viewpoint_camera.timestamp - level_0_trbf_center

    level_0_time_coefficient = gm.t_activation(level_0_tforpoly)

    level_0_rotations = gm.get_rotation(level_0_tforpoly)
    level_0_rotations = level_0_rotations * level_0_time_coefficient
    level_0_colors_precomp = gm.get_features

    level_0_motion = gm.get_motion

    # in test, the means3D, opacities are moved in to cuda: prepreprocessCUDA
    # here we compute for saving and visualization

    level_0_cur_means3D = (
        level_0_means3D
        + level_0_motion[:, 0:3] * level_0_tforpoly * level_0_time_coefficient
        + level_0_motion[:, 3:6] * level_0_tforpoly * level_0_tforpoly * level_0_time_coefficient
        + level_0_motion[:, 6:9] * level_0_tforpoly * level_0_tforpoly * level_0_tforpoly * level_0_time_coefficient
    )
    level_0_velocities3D = (
        level_0_motion[:, 0:3]
        + 2 * level_0_motion[:, 3:6] * level_0_tforpoly
        + 3 * level_0_motion[:, 6:9] * level_0_tforpoly * level_0_tforpoly
    )
    level_0_velocities3D = level_0_velocities3D * level_0_time_coefficient

    level_0_point_opacity = gm.get_opacity

    level_0_trbf_scale = gm.get_trbf_scale

    level_0_trbf_distance = level_0_tforpoly / torch.exp(level_0_trbf_scale)
    level_0_trbf_output = basic_function(level_0_trbf_distance)

    level_0_opacity = level_0_point_opacity * level_0_trbf_output * level_0_time_coefficient

    # computed_opacity is not blend with timestamp
    level_0_computed_opacity = gm.computed_opacity
    level_0_computed_opacity = level_0_computed_opacity * level_0_time_coefficient
    level_0_scales = gm.computed_scales
    level_0_scales = level_0_scales * level_0_time_coefficient
    level_0_computed_trbf_scale = gm.computed_trbf_scale
    level_0_parent_idx_dummy = torch.zeros((n_level_0_points, 1), dtype=torch.long, device="cuda") - 1.0

    if level == 0:
        n_points = n_level_0_points
        cur_means3D = level_0_cur_means3D
        rotations = level_0_rotations
        colors_precomp = level_0_colors_precomp
        opacity = level_0_opacity
        scales = level_0_scales

        means3D = level_0_means3D
        trbf_center = level_0_trbf_center
        trbf_scale = level_0_trbf_scale
        motion = level_0_motion
        velocities3D = level_0_velocities3D
        computed_trbf_scale = level_0_computed_trbf_scale
        computed_opacity = level_0_computed_opacity

        parent_idx = level_0_parent_idx_dummy

    elif level == 1:
        level_1_parent_idx = gm.get_level_1_parent_idx
        n_level_1_points = level_1_parent_idx.shape[0]
        n_points = n_level_0_points + n_level_1_points

        level_1_point_opacity = gm.get_level_1_opacity
        level_1_trbf_center = gm.get_level_1_trbf_center
        level_1_trbf_scale = gm.get_level_1_trbf_scale

        level_1_tforpoly = viewpoint_camera.timestamp - level_1_trbf_center
        if act_level_1:
            level_1_time_coefficient = gm.t_activation(level_1_tforpoly)
        else:
            level_1_time_coefficient = torch.ones_like(level_1_tforpoly)

        level_1_rotations = gm.get_level_1_rotation(level_1_tforpoly)
        level_1_rotations = level_1_rotations * level_1_time_coefficient
        level_1_colors_c = gm.get_level_1_features
        level_1_color_a1 = gm.get_level_1_color_a1
        level_1_color_c1 = gm.get_level_1_color_c1
        level_1_color_a2 = gm.get_level_1_color_a2
        level_1_color_c2 = gm.get_level_1_color_c2
        level_1_color_a3 = gm.get_level_1_color_a3
        level_1_color_c3 = gm.get_level_1_color_c3
        level_1_cubic_color = level_1_colors_c + level_1_color_a1 * (viewpoint_camera.timestamp - level_1_color_c1)
        level_1_cubic_color += level_1_color_a2 * (viewpoint_camera.timestamp - level_1_color_c2) ** 2
        level_1_cubic_color += level_1_color_a3 * (viewpoint_camera.timestamp - level_1_color_c3) ** 3
        level_1_colors_precomp = level_1_cubic_color * level_1_time_coefficient

        level_1_motion = gm.get_level_1_motion

        level_1_parent_idx = level_1_parent_idx.squeeze(1)
        level_1_parent_means3D = level_0_cur_means3D[level_1_parent_idx]
        level_1_parent_velocity3D = level_0_velocities3D[level_1_parent_idx]

        level_1_means3D = level_1_parent_means3D  # used in CUDA

        level_1_delta_means3D = torch.zeros_like(level_1_parent_means3D)
        level_1_delta_velocities3D = torch.zeros_like(level_1_parent_velocity3D)

        level_1_cur_means3D = level_1_parent_means3D + level_1_delta_means3D
        level_1_velocities3D = level_1_parent_velocity3D + level_1_delta_velocities3D

        level_1_point_opacity = gm.get_level_1_opacity
        level_1_trbf_scale = gm.get_level_1_trbf_scale
        level_1_trbf_distance = level_1_tforpoly / torch.exp(level_1_trbf_scale)
        level_1_trbf_output = basic_function(level_1_trbf_distance)
        level_1_opacity = level_1_point_opacity * level_1_trbf_output * level_1_time_coefficient

        level_1_computed_opacity = gm.computed_level_1_opacity
        level_1_computed_opacity = level_1_computed_opacity * level_1_time_coefficient
        level_1_scales = gm.computed_level_1_scales
        level_1_scales = level_1_scales * level_1_time_coefficient
        level_1_computed_trbf_scale = gm.computed_level_1_trbf_scale

        if transp_level_0:
            cur_means3D = level_1_cur_means3D
            rotations = level_1_rotations
            colors_precomp = level_1_colors_precomp
            opacity = level_1_opacity
            scales = level_1_scales
            means3D = level_1_means3D
            trbf_center = level_1_trbf_center
            trbf_scale = level_1_trbf_scale
            motion = level_1_motion
            velocities3D = level_1_velocities3D
            computed_trbf_scale = level_1_computed_trbf_scale
            computed_opacity = level_1_computed_opacity
            parent_idx = level_1_parent_idx

            n_points = n_level_1_points

        else:

            means3D = torch.cat((level_0_means3D, level_1_means3D), dim=0)
            cur_means3D = torch.cat((level_0_cur_means3D, level_1_cur_means3D), dim=0)
            rotations = torch.cat((level_0_rotations, level_1_rotations), dim=0)
            colors_precomp = torch.cat((level_0_colors_precomp, level_1_colors_precomp), dim=0)
            opacity = torch.cat((level_0_opacity, level_1_opacity), dim=0)
            scales = torch.cat((level_0_scales, level_1_scales), dim=0)

            trbf_center = torch.cat((level_0_trbf_center, level_1_trbf_center), dim=0)
            trbf_scale = torch.cat((level_0_trbf_scale, level_1_trbf_scale), dim=0)
            motion = torch.cat((level_0_motion, level_1_motion), dim=0)
            velocities3D = torch.cat((level_0_velocities3D, level_1_velocities3D), dim=0)
            computed_trbf_scale = torch.cat((level_0_computed_trbf_scale, level_1_computed_trbf_scale), dim=0)
            computed_opacity = torch.cat((level_0_computed_opacity, level_1_computed_opacity), dim=0)

            # we use gm.get_level_1_parent_idx instead of level_1_parent_idx, as the dimension is different
            parent_idx = torch.cat((level_0_parent_idx_dummy, gm.get_level_1_parent_idx), dim=0)

    else:
        raise ValueError(f"Invalid level: {level}")

    screen_space_points = torch.zeros((n_points, 3), dtype=means3D.dtype, requires_grad=True, device="cuda") + 0

    means2D = screen_space_points

    point_levels = torch.zeros((n_points, 1), dtype=means3D.dtype, requires_grad=False, device="cuda")

    if transp_level_0:
        # all points are level 1
        point_levels = point_levels + 1
    else:
        point_levels[n_level_0_points:] = 1

    # cuda prepreprocessCUDA will calculate the means3D, opacities with timestamp
    rendered_image, radii = rasterizer(
        timestamp=viewpoint_camera.timestamp,
        trbf_center=trbf_center,
        trbf_scale=computed_trbf_scale,
        motion=motion,
        means3D=means3D,
        means2D=means2D,
        shs=None,
        colors_precomp=colors_precomp,
        opacities=computed_opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
    )

    torch.cuda.synchronize()
    duration = time.time() - start_time
    return {
        "render": rendered_image,
        "trbf_center": trbf_center,
        "trbf_scale": trbf_scale,
        "viewspace_points": screen_space_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "duration": duration,
        "means3D_no_t": means3D,
        "means3D": cur_means3D,
        "means2D": means2D,
        "motion": motion,
        "velocities3D": velocities3D,
        "opacity": opacity,
        "rotations": rotations,
        "colors_precomp": colors_precomp,
        "scales": scales,
        "point_levels": point_levels,
        "parent_idx": parent_idx,
    }


