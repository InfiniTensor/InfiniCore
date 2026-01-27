#ifndef __INFINIOP_CUDA_KERNEL_COMMON_CUH__
#define __INFINIOP_CUDA_KERNEL_COMMON_CUH__

#if defined(ENABLE_HYGON_API)
#define INFINIOP_CUDA_KERNEL __launch_bounds__(1024) __global__ void
#else
#define INFINIOP_CUDA_KERNEL __global__ void
#endif

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

// Posible maximum number of threads per block for CUDA architectures
// Used for picking correct kernel launch configuration
#define CUDA_BLOCK_SIZE_4096 4096
#define CUDA_BLOCK_SIZE_2048 2048
#define CUDA_BLOCK_SIZE_1024 1024
#define CUDA_BLOCK_SIZE_512 512

#define CHECK_CUDA(API) CHECK_INTERNAL(API, cudaSuccess)

#ifdef ENABLE_HYGON_API
// Hygon DCU uses different bfloat16 type definitions
using cuda_bfloat16 = __nv_bfloat16;
using cuda_bfloat162 = __nv_bfloat162;
#else
using cuda_bfloat16 = nv_bfloat16;
using cuda_bfloat162 = nv_bfloat162;
using cuda_fp8_e4m3 = __nv_fp8_e4m3;
#endif

namespace device::nvidia {

// get the memory offset of the given element in a tensor given its flat index
__forceinline__ __device__ __host__ size_t
indexToOffset(
    size_t flat_index,
    size_t ndim,
    const size_t *shape,
    const ptrdiff_t *strides) {
    size_t res = 0;
    for (size_t i = ndim; i-- > 0;) {
        res += (flat_index % shape[i]) * strides[i];
        flat_index /= shape[i];
    }
    return res;
}
} // namespace device::nvidia

__forceinline__ __device__ float
exp_(const float val) {
    return expf(val);
}

#if !defined(ENABLE_ILUVATAR_API) && !defined(ENABLE_QY_API) && !defined(ENABLE_HYGON_API)
__forceinline__ __device__ long double
exp_(const long double val) {
    return expl(val);
}
#endif

__forceinline__ __device__ double
exp_(const double val) {
    return exp(val);
}

__forceinline__ __device__ __half
exp_(const __half x) {
    return hexp(x);
}

__forceinline__ __device__ __nv_bfloat16
exp_(const __nv_bfloat16 x) {
    return hexp(x);
}

#endif // __INFINIOP_CUDA_KERNEL_COMMON_CUH__
