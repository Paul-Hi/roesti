#include "cuda_measurable_event.cuh"
#include "macros.cuh"

CudaMeasurableEvent::CudaMeasurableEvent(const std::string& name)
    : mName(name)
{
    CUDA_CHECK_PRINT(cudaEventCreate(&mStart));
    CUDA_CHECK_PRINT(cudaEventCreate(&mStop));
    CUDA_CHECK_PRINT(cudaEventRecord(mStart));
}

CudaMeasurableEvent::~CudaMeasurableEvent()
{
    CUDA_CHECK_PRINT(cudaEventRecord(mStop));
    CUDA_CHECK_PRINT(cudaEventSynchronize(mStop));
    float milliseconds = 0;
    CUDA_CHECK_PRINT(cudaEventElapsedTime(&milliseconds, mStart, mStop));

    auto& stats = Statistics::get();
    stats.addSample(mName, milliseconds);

    CUDA_CHECK_PRINT(cudaEventDestroy(mStart));
    CUDA_CHECK_PRINT(cudaEventDestroy(mStop));
}
