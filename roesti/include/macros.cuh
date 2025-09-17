#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

#define HD __host__ __device__

#define HOST __host__

#define DEVICE __device__

#define KERNEL __global__

#define DEVICE_CONST __constant__

#define FORCE_INLINE __forceinline__

#ifndef STRINGIFY
#define STRINGIFY(x) #x
#endif

#ifndef STR
#define STR(x) STRINGIFY(x)
#endif

#ifndef FILE_LINE
#define FILE_LINE __FILE__ ":" STR(__LINE__)
#endif

template <typename... Args>
HD std::string __fmt(const std::string& pre, Args&&... args)
{
    char buffer[1024];
    int n = snprintf(buffer, sizeof(buffer), pre.c_str(), std::forward<Args>(args)...);
    return std::string(buffer, n);
}

#ifndef CRITICAL_ASSERT
#define CRITICAL_ASSERT(x, ...)                                                         \
    do                                                                                  \
    {                                                                                   \
        if (!(x))                                                                       \
            throw std::runtime_error(__fmt(std::string(FILE_LINE ":"), ##__VA_ARGS__)); \
    } while (0)
#endif

#ifndef CUDA_CHECK_PRINT
#define CUDA_CHECK_PRINT(x)                                                                        \
    do                                                                                             \
    {                                                                                              \
        cudaError_t result = x;                                                                    \
        if (result != cudaSuccess)                                                                 \
            std::cerr << FILE_LINE ": " #x " failed: " << cudaGetErrorString(result) << std::endl; \
    } while (0)
#endif

#ifndef CUDA_CHECK_THROW
#define CUDA_CHECK_THROW(x)                                                                                          \
    do                                                                                                               \
    {                                                                                                                \
        cudaError_t result = x;                                                                                      \
        if (result != cudaSuccess)                                                                                   \
            throw std::runtime_error(__fmt(std::string(FILE_LINE ": " #x " failed: "), cudaGetErrorString(result))); \
    } while (0)
#endif

#define CHECK_TENSOR(X, DIMS, CHANNELS, TYPE)                                       \
    TORCH_INTERNAL_ASSERT(X.is_cuda(), #X " must be a cuda tensor");                \
    TORCH_INTERNAL_ASSERT(X.scalar_type() == TYPE, #X " must be " #TYPE " tensor"); \
    TORCH_INTERNAL_ASSERT(X.dim() == DIMS, #X " must have " #DIMS " dimensions");   \
    TORCH_INTERNAL_ASSERT(X.size(std::max(0, DIMS - 1)) == CHANNELS, #X " must have " #CHANNELS " channels")

#define CHECK_TENSOR1(X, DIMS, TYPE)                                                \
    TORCH_INTERNAL_ASSERT(X.is_cuda(), #X " must be a cuda tensor");                \
    TORCH_INTERNAL_ASSERT(X.scalar_type() == TYPE, #X " must be " #TYPE " tensor"); \
    TORCH_INTERNAL_ASSERT(X.dim() == DIMS, #X " must have " #DIMS " dimensions");
