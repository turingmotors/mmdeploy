// Copyright (c) OpenMMLab. All rights reserved.
#ifndef TRT_GRID_SAMPLER_KERNEL_HPP
#define TRT_GRID_SAMPLER_KERNEL_HPP
#include <cuda_runtime.h>

enum class GridSamplerInterpolation { Bilinear, Nearest };
enum class GridSamplerPadding { Zeros, Border, Reflection };

template <typename T>
void grid_sample(T *output, const T *input, const T *grid, int64_t *output_dims, int64_t *input_dims,
                 int64_t *grid_dims, int nb_dims, GridSamplerInterpolation interp,
                 GridSamplerPadding padding, bool align_corners, cudaStream_t stream);
#endif  // TRT_GRID_SAMPLER_KERNEL_HPP
