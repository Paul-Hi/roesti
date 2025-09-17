#pragma once

#include <cuda_surface_types.h>
#include <torch/torch.h>

void copyToCudaSurface(const torch::Tensor& colorBuffer, cudaSurfaceObject_t cudaSurface, int width, int height);
