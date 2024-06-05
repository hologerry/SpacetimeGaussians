import math
import time

import torch

from gaussian_splatting.gaussian.gm_simple_color_scale_rotation_act_two_sp_level import GaussianModel


def train_lite_act_two_sp_level(
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

    means3D = gm.get_xyz if level == 0 else gm.get_all_xyz

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screen_space_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda") + 0
    point_times = torch.ones((means3D.shape[0], 1), dtype=means3D.dtype, requires_grad=False, device="cuda") + 0
    try:
        screen_space_points.retain_grad()
    except:
        pass

    means2D = screen_space_points

    point_opacity = gm.get_opacity if level == 0 else gm.get_all_opacity

    trbf_center = gm.get_trbf_center if level == 0 else gm.get_all_trbf_center
    trbf_scale = gm.get_trbf_scale if level == 0 else gm.get_all_trbf_scale

    trbf_distance_offset = viewpoint_camera.timestamp * point_times - trbf_center

    time_coefficient = gm.t_activation(trbf_distance_offset)
    if not act_level_1:
        n_level_0_points = gm.get_xyz.shape[0]
        time_coefficient[n_level_0_points:] = 1.0

    trbf_distance = trbf_distance_offset / torch.exp(trbf_scale)
    trbf_output = basic_function(trbf_distance)

    opacity = point_opacity * trbf_output * time_coefficient
    if level == 0:
        gm.trbf_output = trbf_output
    else:
        gm.all_trbf_output = trbf_output

    scales = gm.get_scaling if level == 0 else gm.get_all_scaling
    scales = scales * time_coefficient

    motion = gm.get_motion if level == 0 else gm.get_all_motion

    tforpoly = trbf_distance_offset.detach()
    cur_means3D = (
        means3D
        + motion[:, 0:3] * tforpoly * time_coefficient
        + motion[:, 3:6] * tforpoly * tforpoly * time_coefficient
        + motion[:, 6:9] * tforpoly * tforpoly * tforpoly * time_coefficient
    )

    rotations = gm.get_rotation(tforpoly) if level == 0 else gm.get_all_rotation(tforpoly)
    rotations = rotations * time_coefficient
    colors_precomp = gm.get_features if level == 0 else gm.get_all_features
    colors_precomp = colors_precomp * time_coefficient

    cov3D_precomp = None

    shs = None

    rendered_image, radii, depth = rasterizer(
        means3D=cur_means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    return {
        "render": rendered_image,
        "viewspace_points": screen_space_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "opacity": opacity,
        "depth": depth,
    }



def test_lite_act_two_sp_level_vis(
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

    means3D = gm.get_xyz if level == 0 else gm.get_all_xyz

    screen_space_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda") + 0

    trbf_center = gm.get_trbf_center if level == 0 else gm.get_all_trbf_center
    tforpoly = viewpoint_camera.timestamp - trbf_center

    time_coefficient = gm.t_activation(tforpoly)
    if not act_level_1:
        n_level_0_points = gm.get_xyz.shape[0]
        time_coefficient[n_level_0_points:] = 1.0

    rotations = gm.get_rotation(tforpoly) if level == 0 else gm.get_all_rotation(tforpoly)
    rotations = rotations * time_coefficient
    colors_precomp = gm.get_features if level == 0 else gm.get_all_features

    motion = gm.get_motion if level == 0 else gm.get_all_motion

    # in test, the means3D, opacities are moved in to cuda: prepreprocessCUDA
    # here we compute for saving and visualization

    cur_means3D = (
        means3D
        + motion[:, 0:3] * tforpoly * time_coefficient
        + motion[:, 3:6] * tforpoly * tforpoly * time_coefficient
        + motion[:, 6:9] * tforpoly * tforpoly * tforpoly * time_coefficient
    )
    velocities3D = motion[:, 0:3] + 2 * motion[:, 3:6] * tforpoly + 3 * motion[:, 6:9] * tforpoly * tforpoly
    velocities3D = velocities3D * time_coefficient

    point_opacity = gm.get_opacity if level == 0 else gm.get_all_opacity

    trbf_scale = gm.get_trbf_scale if level == 0 else gm.get_all_trbf_scale

    trbf_distance = tforpoly / torch.exp(trbf_scale)
    trbf_output = basic_function(trbf_distance)

    opacity = point_opacity * trbf_output * time_coefficient

    # computed_opacity is not blend with timestamp
    computed_opacity = gm.computed_opacity if level == 0 else gm.computed_all_opacity
    computed_opacity = computed_opacity * time_coefficient
    scales = gm.computed_scales if level == 0 else gm.computed_all_scales
    scales = scales * time_coefficient
    computed_trbf_scale = gm.computed_trbf_scale if level == 0 else gm.computed_all_trbf_scale

    means2D = screen_space_points

    cov3D_precomp = None

    shs = None

    point_levels = torch.zeros((means3D.shape[0], 1), dtype=means3D.dtype, requires_grad=False, device="cuda")

    n_level_0 = gm.get_xyz.shape[0]
    point_levels[:n_level_0] = 0
    point_levels[n_level_0:] = 1

    # cuda prepreprocessCUDA will calculate the means3D, opacities with timestamp
    rendered_image, radii = rasterizer(
        timestamp=viewpoint_camera.timestamp,
        trbf_center=trbf_center,
        trbf_scale=computed_trbf_scale,
        motion=motion,
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=computed_opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
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
    }
