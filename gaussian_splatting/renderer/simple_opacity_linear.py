import math
import time

import torch

from gaussian_splatting.scene.ours_simple_opacity_linear import GaussianModel


def train_ours_lite_opacity_linear(
    viewpoint_camera,
    gm: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    basic_function=None,
    GRsetting=None,
    GRzer=None,
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screen_space_points = torch.zeros_like(gm.get_xyz, dtype=gm.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    point_times = torch.ones((gm.get_xyz.shape[0], 1), dtype=gm.get_xyz.dtype, requires_grad=False, device="cuda") + 0
    try:
        screen_space_points.retain_grad()
    except:
        pass

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

    means3D = gm.get_xyz
    means2D = screen_space_points

    trbf_center = gm.get_trbf_center
    # trbf_scale = gm.get_trbf_scale # no trbf_scale in linear

    trbf_distance_offset = viewpoint_camera.timestamp * point_times - trbf_center

    point_opacity = gm.get_opacity(trbf_distance_offset)
    trbf_output = trbf_distance_offset  # * trbf_scale

    opacity = point_opacity  # * trbf_output  # - 0.5
    gm.trbf_output = trbf_output

    scales = gm.get_scaling

    tforpoly = trbf_distance_offset.detach()
    means3D = (
        means3D
        + gm._motion[:, 0:3] * tforpoly
        + gm._motion[:, 3:6] * tforpoly * tforpoly
        + gm._motion[:, 6:9] * tforpoly * tforpoly * tforpoly
    )

    rotations = gm.get_rotation(tforpoly)  # to try use
    colors_precomp = gm.get_features(tforpoly)

    cov3D_precomp = None

    shs = None

    rendered_image, radii, depth = rasterizer(
        means3D=means3D,
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


def test_ours_lite_opacity_linear_vis(
    viewpoint_camera,
    gm: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    basic_function=None,
    GRsetting=None,
    GRzer=None,
):

    screen_space_points = torch.zeros_like(gm.get_xyz, dtype=gm.get_xyz.dtype, requires_grad=True, device="cuda") + 0

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

    tforpoly = viewpoint_camera.timestamp - gm.get_trbf_center

    rotations = gm.get_rotation(tforpoly)  # to try use
    colors_precomp = gm.get_features(tforpoly)

    motion = gm._motion

    means3D = gm.get_xyz

    # in test, the means3D, opacities are moved in to cuda: prepreprocessCUDA
    # here we compute for saving and visualization

    means3D = (
        means3D
        + motion[:, 0:3] * tforpoly
        + motion[:, 3:6] * tforpoly * tforpoly
        + motion[:, 6:9] * tforpoly * tforpoly * tforpoly
    )
    velocities3D = motion[:, 0:3] + 2 * motion[:, 3:6] * tforpoly + 3 * motion[:, 6:9] * tforpoly * tforpoly

    point_opacity = gm.get_opacity(tforpoly)

    # trbf_scale = gm.get_trbf_scale

    # trbf_distance = tforpoly / torch.exp(trbf_scale)
    # trbf_output = basic_function(trbf_distance) # in exp linear the basic function is different

    opacity = point_opacity  # * trbf_output  # - 0.5

    # computed_opacity is not blend with timestamp
    computed_opacity = gm.computed_opacity
    scales = gm.computed_scales

    means2D = screen_space_points

    cov3D_precomp = None

    shs = None

    # cuda prepreprocessCUDA will calculate the means3D, opacities with timestamp
    rendered_image, radii = rasterizer(
        timestamp=viewpoint_camera.timestamp,
        trbf_center=gm.get_trbf_center,
        trbf_scale=gm.computed_trbf_scale,
        motion=gm._motion,
        means3D=gm.get_xyz,
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
        "trbf_center": gm.get_trbf_center,
        "trbf_scale": gm.get_trbf_scale,
        "viewspace_points": screen_space_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "duration": duration,
        "means3D_no_t": gm.get_xyz,
        "means3D": means3D,
        "means2D": means2D,
        "motion": motion,
        "velocities3D": velocities3D,
        "opacity": opacity,
        "rotations": rotations,
        "colors_precomp": colors_precomp,
        "scales": scales,
    }
