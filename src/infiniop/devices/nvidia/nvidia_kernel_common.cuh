#ifndef __INFINIOP_CUDA_KERNEL_COMMON_CUH__
#define __INFINIOP_CUDA_KERNEL_COMMON_CUH__

#if defined(ENABLE_HYGON_API)
#define INFINIOP_CUDA_KERNEL __launch_bounds__(1024) __global__ void
#else
#define INFINIOP_CUDA_KERNEL __global__ void
#endif

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#ifndef ENABLE_HYGON_API
#include <cuda_fp8.h>
#endif

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

// Store FP8 E4M3FN values as raw bytes in portable CUDA-style kernels.
// Several CUDA-compatible backends either expose a different FP8 type name or
// do not ship cuda_fp8.h at all, while the tensor ABI is still one byte.
__forceinline__ __device__ uint8_t
infiniopFp8E4m3Encode(float value) {
    const uint32_t bits = __float_as_uint(value);
    const uint32_t abs_bits = bits & 0x7fffffffU;
    uint32_t magnitude = 0x7fU;
    if (abs_bits < 0x43f00000U) {
        if (abs_bits > 0x3c7fffffU) {
            const uint32_t tie = (bits >> 20U) & 1U;
            magnitude = (bits + tie + 0x0407ffffU) >> 20U;
        } else {
            const float absolute = __uint_as_float(abs_bits);
            magnitude = __float_as_uint(absolute + 16384.0f);
        }
    }
    const uint32_t sign = (bits >> 24U) & 0x80U;
    return static_cast<uint8_t>((magnitude & 0x7fU) | sign);
}

__forceinline__ __device__ float
infiniopFp8E4m3Decode(uint8_t value) {
    const uint32_t magnitude = value & 0x7fU;
    const uint32_t exponent = magnitude >> 3U;
    const uint32_t mantissa = magnitude & 0x7U;
    if (exponent == 0xfU && mantissa == 0x7U) {
        return __uint_as_float(0x7fffffffU);
    }
    const float decoded = exponent == 0U
                            ? static_cast<float>(mantissa) * 0.001953125f
                            : __uint_as_float((exponent + 120U) << 23U)
                                  * (1.0f + static_cast<float>(mantissa) * 0.125f);
    return (value & 0x80U) != 0U ? -decoded : decoded;
}

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

using device::nvidia::indexToOffset;

__forceinline__ __device__ float
exp_(const float val) {
    return expf(val);
}

#if !defined(ENABLE_ILUVATAR_API) && !defined(ENABLE_QY_API) && !defined(ENABLE_HYGON_API) && !defined(ENABLE_ALI_API)
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
