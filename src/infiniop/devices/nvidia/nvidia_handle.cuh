#ifndef __INFINIOP_CUDA_HANDLE_CUH__
#define __INFINIOP_CUDA_HANDLE_CUH__

#include "../../../utils.h"
#include "../pool.h"
#include "nvidia_handle.h"
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp8.h>
#include <functional>

#ifdef ENABLE_CUDNN_API
#include <cudnn.h>
#endif

#ifdef ENABLE_CUBLASLT_API
#include <cublasLt.h>
#if CUDA_VERSION >= 12090
#define SUPPORT_FP8_BLOCKWISE_SCALE 1
#else
#define SUPPORT_FP8_BLOCKWISE_SCALE 0
#endif
#endif

#define CHECK_CUBLAS(API) CHECK_INTERNAL(API, CUBLAS_STATUS_SUCCESS)
#define CHECK_CUDNN(API) CHECK_INTERNAL(API, CUDNN_STATUS_SUCCESS)
#define CHECK_CUBLASLT(API) CHECK_INTERNAL(API, CUBLAS_STATUS_SUCCESS)

namespace device::nvidia {

class Handle::Internal {
    Pool<cublasHandle_t> blas_handles;
#ifdef ENABLE_CUDNN_API
    Pool<cudnnHandle_t> dnn_handles;
#endif
#ifdef ENABLE_CUBLASLT_API
    Pool<cublasLtHandle_t> blaslt_handles;
#endif

    int _warp_size,
        _max_threads_per_block,
        _block_size[3],
        _grid_size[3];

    template <typename T>
    using Fn = std::function<infiniStatus_t(T)>;

public:
    Internal(int);

    infiniStatus_t useCublas(cudaStream_t stream, const Fn<cublasHandle_t> &f) const;
#ifdef ENABLE_CUDNN_API
    infiniStatus_t useCudnn(cudaStream_t stream, const Fn<cudnnHandle_t> &f) const;
#endif
#ifdef ENABLE_CUBLASLT_API
    infiniStatus_t useCublasLt(cudaStream_t stream, const Fn<cublasLtHandle_t> &f) const;
#endif

    int warpSize() const;
    int maxThreadsPerBlock() const;
    int blockSizeX() const;
    int blockSizeY() const;
    int blockSizeZ() const;
    int gridSizeX() const;
    int gridSizeY() const;
    int gridSizeZ() const;
};

} // namespace device::nvidia

#endif // __INFINIOP_CUDA_HANDLE_CUH__
