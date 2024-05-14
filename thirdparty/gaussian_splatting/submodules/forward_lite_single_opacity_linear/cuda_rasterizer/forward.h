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

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace FORWARD {
// Perform initial steps for each Gaussian prior to rasterization.
void prepreprocess(
    int P,
    const float timestamp,
    const float *trbf_center,
    const float *trbf_scale,
    const float *motion,
    const float *orig_points,
    float *orig_pointsdummy,
    const float *opacities,
    float *opacitiesdummy);

void preprocess(
    int P, int D, int M,
    float *orig_points,
    const glm::vec3 *scales,
    const float scale_modifier,
    const glm::vec4 *rotations,
    float *opacities,
    const float *shs,
    bool *clamped,
    const float *cov3D_precomp,
    const float *colors_precomp,
    const float *view_matrix,
    const float *proj_matrix,
    const glm::vec3 *cam_pos,
    const int W, int H,
    const float focal_x, float focal_y,
    const float tan_fov_x, float tan_fov_y,
    int *radii,
    float2 *points_xy_image,
    float *depths,
    float *cov3Ds,
    float *colors,
    float4 *conic_opacity,
    const dim3 grid,
    uint32_t *tiles_touched,
    bool prefiltered);

// Main rasterization method.
void render(
    const dim3 grid, dim3 block,
    const uint2 *ranges,
    const uint32_t *point_list,
    int W, int H,
    const float2 *points_xy_image,
    const float *features,
    const float4 *conic_opacity,
    float *final_T,
    uint32_t *n_contrib,
    const float *bg_color,
    float *out_color);
} // namespace FORWARD

#endif