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

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <functional>
#include <vector>

// const float timestamp,
// const torch::Tensor& trbf_center,
// const torch::Tensor& trbf_scale,
// const torch::Tensor& motion,

namespace CudaRasterizer {
class Rasterizer {
public:
    static void markVisible(
        int P,
        float *means3D,
        float *view_matrix,
        float *proj_matrix,
        bool *present);

    // std::function<char* (size_t)> geometryBuffer,
    // std::function<char* (size_t)> binningBuffer,
    // std::function<char* (size_t)> imageBuffer,
    // const int P, int D, int M,
    // const float timestamp,
    // const float* trbf_center,
    // const float* trbf_scale,
    // const float* motion,
    // const float* means3D,
    // const float* opacities,
    // const float* background,
    // const int width, int height,
    // const float* means3Ddummy,
    // const float* shs,
    // const float* colors_precomp,
    // const float* opacitiesdummy,
    // const float* scales,
    // const float scale_modifier,
    // const float* rotations,
    // const float* cov3D_precomp,
    // const float* view_matrix,
    // const float* proj_matrix,
    // const float* cam_pos,
    // const float tan_fov_x, float tan_fov_y,
    // const bool prefiltered,
    // float* out_color,
    // int* radii,
    // bool debug

    static int forward(
        std::function<char *(size_t)> geometryBuffer,
        std::function<char *(size_t)> binningBuffer,
        std::function<char *(size_t)> imageBuffer,
        const int P, int D, int M,
        const float timestamp,
        const float *trbf_center,
        const float *trbf_scale,
        const float *motion,
        const float *means3D,
        float *means3Ddummy,
        const float *opacities,
        float *opacitiesdummy,
        const float *background,
        const int width, int height,
        const float *shs,
        const float *colors_precomp,
        const float *scales,
        const float scale_modifier,
        const float *rotations,
        const float *cov3D_precomp,
        const float *view_matrix,
        const float *proj_matrix,
        const float *cam_pos,
        const float tan_fov_x, float tan_fov_y,
        const bool prefiltered,
        float *out_color,
        int *radii = nullptr,
        bool debug = false);

    static void backward(
        const int P, int D, int M, int R,
        const float *background,
        const int width, int height,
        const float *means3D,
        const float *shs,
        const float *colors_precomp,
        const float *scales,
        const float scale_modifier,
        const float *rotations,
        const float *cov3D_precomp,
        const float *view_matrix,
        const float *proj_matrix,
        const float *campos,
        const float tan_fov_x, float tan_fov_y,
        const int *radii,
        char *geom_buffer,
        char *binning_buffer,
        char *image_buffer,
        const float *dL_dpix,
        float *dL_dmean2D,
        float *dL_dconic,
        float *dL_dopacity,
        float *dL_dcolor,
        float *dL_dmean3D,
        float *dL_dcov3D,
        float *dL_dsh,
        float *dL_dscale,
        float *dL_drot,
        bool debug);
};
}; // namespace CudaRasterizer

#endif
