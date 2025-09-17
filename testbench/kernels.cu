#include "kernels.cuh"
#include <cuda_runtime.h>

__global__ void toCudaSurfaceKernel(const float4* __restrict__ colorBuffer, cudaSurfaceObject_t cudaSurface, int width, int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        const int index = y * width + x;
        const float4& f = colorBuffer[index];

        surf2Dwrite(f, cudaSurface, x * sizeof(float4), y);
    }
}

void copyToCudaSurface(const torch::Tensor& colorBuffer, cudaSurfaceObject_t cudaSurface, int width, int height)
{
    const dim3 blockSize(16, 16);
    const dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    toCudaSurfaceKernel<<<gridSize, blockSize>>>(
        reinterpret_cast<const float4*>(colorBuffer.contiguous().data_ptr<float>()),
        cudaSurface,
        width,
        height);
}
