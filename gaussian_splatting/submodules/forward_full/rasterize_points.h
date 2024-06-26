/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
// args = (
//     timestamp,
//     trbf_center,
//     trbf_scale,
//     motion,
//     raster_settings.bg,
//     means3D,
//     colors_precomp,
//     opacities,
//     scales,
//     rotations,
//     raster_settings.scale_modifier,
//     cov3Ds_precomp,
//     raster_settings.view_matrix,
//     raster_settings.proj_matrix,
//     raster_settings.tan_fov_x,
//     raster_settings.tan_fov_y,
//     raster_settings.image_height,
//     raster_settings.image_width,
//     sh,
//     raster_settings.sh_degree,
//     raster_settings.campos,
//     raster_settings.prefiltered,
//     raster_settings.debug
// )

RasterizeGaussiansCUDA(
	const float timestamp,
	const torch::Tensor& trbf_center,
	const torch::Tensor& trbf_scale,
	const torch::Tensor& motion,
	const torch::Tensor& background,
	const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& view_matrix,
	const torch::Tensor& proj_matrix,
	const torch::Tensor& mlp1,
	const torch::Tensor& mlp2,
	const torch::Tensor& rayimage,
	const float tan_fov_x,
	const float tan_fov_y,
    const int image_height,
    const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool debug);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
 	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
    const torch::Tensor& colors,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& view_matrix,
    const torch::Tensor& proj_matrix,
	const float tan_fov_x,
	const float tan_fov_y,
    const torch::Tensor& dL_dout_color,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool debug);

torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& view_matrix,
		torch::Tensor& proj_matrix);