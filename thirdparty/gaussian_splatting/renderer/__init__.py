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


import math
import time

import torch
import torch.nn.functional as F

from thirdparty.gaussian_splatting.scene.ours_full import GaussianModel
from thirdparty.gaussian_splatting.utils.graphics_utils import (
    focal2fov,
    fov2focal,
    get_projection_matrix_cv,
)
from thirdparty.gaussian_splatting.utils.sh_utils import eval_sh


def train_ours_full(
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
    # point times just ones, used for board casting
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
    point_opacity = gm.get_opacity

    trbf_center = gm.get_trbf_center
    trbf_scale = gm.get_trbf_scale

    trbf_distance_offset = viewpoint_camera.timestamp * point_times - trbf_center
    trbf_distance = trbf_distance_offset / torch.exp(trbf_scale)
    trbf_output = basic_function(trbf_distance)

    opacity = point_opacity * trbf_output  # - 0.5
    gm.trbf_output = trbf_output

    cov3D_precomp = None

    scales = gm.get_scaling
    shs = None
    tforpoly = trbf_distance_offset.detach()  # Polynomial Motion Trajectory.
    means3D = (
        means3D
        + gm._motion[:, 0:3] * tforpoly
        + gm._motion[:, 3:6] * tforpoly * tforpoly
        + gm._motion[:, 6:9] * tforpoly * tforpoly * tforpoly
    )
    rotations = gm.get_rotation(tforpoly)  # to try use
    colors_precomp = gm.get_features(tforpoly)
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

    rendered_image = gm.rgb_decoder(rendered_image.unsqueeze(0), viewpoint_camera.rays, viewpoint_camera.timestamp)
    rendered_image = rendered_image.squeeze(0)
    return {
        "render": rendered_image,
        "viewspace_points": screen_space_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "opacity": opacity,
        "depth": depth,
    }


def test_ours_full(
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
    torch.cuda.synchronize()
    start_time = time.time()

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
    point_opacity = gm.get_opacity

    trbf_center = gm.get_trbf_center
    trbf_scale = gm.get_trbf_scale

    trbf_distance_offset = viewpoint_camera.timestamp * point_times - trbf_center
    trbf_distance = trbf_distance_offset / torch.exp(trbf_scale)
    trbf_output = basic_function(trbf_distance)

    opacity = point_opacity * trbf_output  # - 0.5
    gm.trbf_output = trbf_output

    cov3D_precomp = None

    scales = gm.get_scaling
    shs = None
    tforpoly = trbf_distance_offset.detach()
    means3D = (
        means3D
        + gm._motion[:, 0:3] * tforpoly
        + gm._motion[:, 3:6] * tforpoly * tforpoly
        + gm._motion[:, 6:9] * tforpoly * tforpoly * tforpoly
    )
    rotations = gm.get_rotation(tforpoly)  # to try use
    colors_precomp = gm.get_features(tforpoly)
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

    rendered_image = gm.rgb_decoder(
        rendered_image.unsqueeze(0), viewpoint_camera.rays, viewpoint_camera.timestamp
    )  # 1 , 3
    rendered_image = rendered_image.squeeze(0)
    torch.cuda.synchronize()
    duration = time.time() - start_time

    return {
        "render": rendered_image,
        "viewspace_points": screen_space_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "opacity": opacity,
        "depth": depth,
        "duration": duration,
    }


def train_ours_lite(
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

    point_opacity = gm.get_opacity

    trbf_center = gm.get_trbf_center
    trbf_scale = gm.get_trbf_scale

    trbf_distance_offset = viewpoint_camera.timestamp * point_times - trbf_center
    trbf_distance = trbf_distance_offset / torch.exp(trbf_scale)
    trbf_output = basic_function(trbf_distance)

    opacity = point_opacity * trbf_output  # - 0.5
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


def test_ours_lite(
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

    means2D = screen_space_points

    cov3D_precomp = None

    shs = None

    # in test, the means3D, opacities are moved in to cuda: prepreprocessCUDA

    rendered_image, radii = rasterizer(
        timestamp=viewpoint_camera.timestamp,
        trbf_center=gm.get_trbf_center,
        trbf_scale=gm.computed_trbf_scale,
        motion=gm._motion,
        means3D=gm.get_xyz,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=gm.computed_opacity,
        scales=gm.computed_scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    torch.cuda.synchronize()
    duration = time.time() - start_time
    return {
        "render": rendered_image,
        "viewspace_points": screen_space_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "duration": duration,
    }


def test_ours_lite_vis(
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

    point_opacity = gm.get_opacity

    trbf_scale = gm.get_trbf_scale

    trbf_distance = tforpoly / torch.exp(trbf_scale)
    trbf_output = basic_function(trbf_distance)

    opacity = point_opacity * trbf_output  # - 0.5

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


def train_ours_full_ss(
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
    point_opacity = gm.get_opacity

    trbf_center = gm.get_trbf_center
    trbf_scale = gm.get_trbf_scale

    trbf_distance_offset = viewpoint_camera.timestamp * point_times - trbf_center
    trbf_distance = trbf_distance_offset / torch.exp(trbf_scale)
    trbf_output = basic_function(trbf_distance)
    opacity = point_opacity * trbf_output  # - 0.5

    gm.trbf_output = trbf_output
    cov3D_precomp = None

    scales = gm.get_scaling

    shs = None

    tforpoly = trbf_distance_offset.detach()

    means3D = (
        means3D
        + gm._motion[:, 0:3] * tforpoly
        + gm._motion[:, 3:6] * tforpoly * tforpoly
        + gm._motion[:, 6:9] * tforpoly * tforpoly * tforpoly
    )

    rotations = gm.get_rotation(tforpoly)

    colors_precomp = gm.get_features(tforpoly)
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

    raw_rendered_image = gm.rgb_decoder(rendered_image.unsqueeze(0), viewpoint_camera.rays)  # 1 , 3
    new_image = F.grid_sample(
        raw_rendered_image, viewpoint_camera.fisheye_mapper, mode="bicubic", padding_mode="zeros", align_corners=True
    )
    # INTER_CUBIC
    rendered_image = F.interpolate(
        new_image,
        size=(int(0.5 * viewpoint_camera.image_height), int(0.5 * viewpoint_camera.image_width)),
        mode="bicubic",
        align_corners=True,
    )
    # rendered_image = torch.clamp(rendered_image, min=0.0, max=1.0)#

    rendered_image = rendered_image.squeeze(0)
    return {
        "render": rendered_image,
        "viewspace_points": screen_space_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "opacity": opacity,
        "raw_rendered_image": raw_rendered_image,
        "depth": depth,
    }


def test_ours_full_ss_fused(
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

    gm.rgb_decoder.mlp2.weight.shape (3,6,1,1)

    gm.rgb_decoder.mlp2.weight.flatten()[0:6] = gm.rgb_decoder.mlp2.weight[0]

    # https://stackoverflow.com/questions/76949152/why-does-pytorchs-linear-layer-store-the-weight-in-shape-out-in-and-transpos
    # store in (out_feature, in_feature) order..
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screen_space_points = torch.zeros_like(gm.get_xyz, dtype=gm.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    # mapper_ss = viewpoint_camera.fisheye_mapper.permute(0,3,1,2)
    # mapper_origin_size =  F.interpolate(mapper_ss, size=(int(0.5*viewpoint_camera.image_height) , int(0.5*viewpoint_camera.image_width)), mode='bicubic', align_corners=True)
    # mapper_origin_size = mapper_origin_size.permute(0,2,3,1)

    torch.cuda.synchronize()
    start_time = time.time()  # perf_counter

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

    means2D = screen_space_points

    cov3D_precomp = None

    # scales = gm.get_scaling

    shs = None
    # sandwich_old 2,3
    rendered_image, radii = rasterizer(
        timestamp=viewpoint_camera.timestamp,
        trbf_center=gm.get_trbf_center,
        trbf_scale=gm.computed_trbf_scale,
        motion=gm._motion,
        means3D=gm.get_xyz,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=gm.computed_opacity,
        scales=gm.computed_scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
        mlp1=gm.rgb_decoder.mlp2.weight.squeeze(3).squeeze(2).flatten(),
        mlp2=gm.rgb_decoder.mlp3.weight.squeeze(3).squeeze(2).flatten(),
        ray_image=viewpoint_camera.rays,
    )

    torch.cuda.synchronize()
    duration = time.time() - start_time  #

    rendered_image = F.grid_sample(
        rendered_image.unsqueeze(0),
        viewpoint_camera.fisheye_mapper,
        mode="bicubic",
        padding_mode="zeros",
        align_corners=True,
    )
    # INTER_CUBIC
    rendered_image = F.interpolate(
        rendered_image,
        size=(int(0.5 * viewpoint_camera.image_height), int(0.5 * viewpoint_camera.image_width)),
        mode="bicubic",
        align_corners=True,
    )
    rendered_image = rendered_image.squeeze(0)

    # rendered_image = F.grid_sample(rendered_image.unsqueeze(0), mapper_origin_size, mode='bicubic', padding_mode='zeros', align_corners=True)
    #  raw_rendered_image = gm.rgb_decoder(rendered_image.unsqueeze(0), viewpoint_camera.rays) # 1 , 3
    ## rendered_image = rendered_image #[0:3,:,:]
    # rendered_image = torch.sigmoid(rendered_image) # sigmoid not fused.
    # rendered_image = rendered_image.squeeze(0)
    return {
        "render": rendered_image,
        "viewspace_points": screen_space_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "raw_rendered_image": rendered_image,
        "duration": duration,
    }


def test_ours_full_ss(
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
    point_times = (
        torch.ones((gm.get_xyz.shape[0], 1), dtype=gm.get_xyz.dtype, requires_grad=False, device="cuda") + 0
    )  #
    try:
        screen_space_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    torch.cuda.synchronize()
    start_time = time.time()  # perf_counter

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
    point_opacity = gm.get_opacity

    trbf_center = gm.get_trbf_center
    trbf_scale = gm.get_trbf_scale

    trbf_distance_offset = viewpoint_camera.timestamp * point_times - trbf_center
    trbf_distance = trbf_distance_offset / torch.exp(trbf_scale)
    trbf_output = basic_function(trbf_distance)
    opacity = point_opacity * trbf_output  # - 0.5

    gm.trbf_output = trbf_output
    cov3D_precomp = None

    scales = gm.get_scaling

    shs = None

    tforpoly = trbf_distance_offset.detach()

    means3D = (
        means3D
        + gm._motion[:, 0:3] * tforpoly
        + gm._motion[:, 3:6] * tforpoly * tforpoly
        + gm._motion[:, 6:9] * tforpoly * tforpoly * tforpoly
    )

    rotations = gm.get_rotation(tforpoly)

    colors_precomp = gm.get_features(tforpoly)
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

    raw_rendered_image = gm.rgb_decoder(rendered_image.unsqueeze(0), viewpoint_camera.rays)  # 1 , 3
    new_image = F.grid_sample(
        raw_rendered_image, viewpoint_camera.fisheye_mapper, mode="bicubic", padding_mode="zeros", align_corners=True
    )
    # INTER_CUBIC
    rendered_image = F.interpolate(
        new_image,
        size=(int(0.5 * viewpoint_camera.image_height), int(0.5 * viewpoint_camera.image_width)),
        mode="bicubic",
        align_corners=True,
    )

    torch.cuda.synchronize()
    duration = time.time() - start_time

    rendered_image = rendered_image.squeeze(0)
    return {
        "render": rendered_image,
        "viewspace_points": screen_space_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "opacity": opacity,
        "raw_rendered_image": raw_rendered_image,
        "depth": depth,
        "duration": duration,
    }


def train_ours_lite_ss(
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
    point_opacity = gm.get_opacity

    trbf_center = gm.get_trbf_center
    trbf_scale = gm.get_trbf_scale

    trbf_distance_offset = viewpoint_camera.timestamp * point_times - trbf_center
    trbf_distance = trbf_distance_offset / torch.exp(trbf_scale)
    trbf_output = basic_function(trbf_distance)

    opacity = point_opacity * trbf_output  # - 0.5
    gm.trbf_output = trbf_output

    cov3D_precomp = None

    scales = gm.get_scaling

    shs = None
    tforpoly = trbf_distance_offset.detach()
    means3D = (
        means3D
        + gm._motion[:, 0:3] * tforpoly
        + gm._motion[:, 3:6] * tforpoly * tforpoly
        + gm._motion[:, 6:9] * tforpoly * tforpoly * tforpoly
    )

    rotations = gm.get_rotation(tforpoly)  # to try use
    colors_precomp = gm.get_features(tforpoly)

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

    raw_rendered_image = rendered_image.unsqueeze(0)
    new_image = F.grid_sample(
        raw_rendered_image, viewpoint_camera.fisheye_mapper, mode="bicubic", padding_mode="zeros", align_corners=True
    )
    # INTER_CUBIC
    rendered_image = F.interpolate(
        new_image,
        size=(int(0.5 * viewpoint_camera.image_height), int(0.5 * viewpoint_camera.image_width)),
        mode="bicubic",
        align_corners=True,
    )
    # rendered_image = torch.clamp(rendered_image, min=0.0, max=1.0)#

    rendered_image = rendered_image.squeeze(0)
    return {
        "render": rendered_image,
        "viewspace_points": screen_space_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "opacity": opacity,
        "raw_rendered_image": raw_rendered_image,
        "depth": depth,
    }


def test_ours_lite_ss(
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

    means2D = screen_space_points

    cov3D_precomp = None

    shs = None

    rendered_image, radii = rasterizer(
        timestamp=viewpoint_camera.timestamp,
        trbf_center=gm.get_trbf_center,
        trbf_scale=gm.computed_trbf_scale,
        motion=gm._motion,
        means3D=gm.get_xyz,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=gm.computed_opacity,
        scales=gm.computed_scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    torch.cuda.synchronize()
    duration = time.time() - start_time  #

    rendered_image = F.grid_sample(
        rendered_image.unsqueeze(0),
        viewpoint_camera.fisheye_mapper,
        mode="bicubic",
        padding_mode="zeros",
        align_corners=True,
    )
    # INTER_CUBIC
    rendered_image = F.interpolate(
        rendered_image,
        size=(int(0.5 * viewpoint_camera.image_height), int(0.5 * viewpoint_camera.image_width)),
        mode="bicubic",
        align_corners=True,
    )
    rendered_image = rendered_image.squeeze(0)

    return {
        "render": rendered_image,
        "viewspace_points": screen_space_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "duration": duration,
    }
