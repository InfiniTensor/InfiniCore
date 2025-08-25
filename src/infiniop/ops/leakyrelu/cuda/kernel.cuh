#ifndef __LEAKYRELU_CUDA_KERNEL_CUH__
#define __LEAKYRELU_CUDA_KERNEL_CUH__

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <type_traits>

template <typename DevT>
__device__ __forceinline__ float to_float_for_leaky(const DevT &v) {
    if constexpr (std::is_same_v<DevT, half>) {
        return __half2float(v);
    } else if constexpr (std::is_same_v<DevT, __nv_bfloat16>) {
        return __bfloat162float(v);
    } else {
        return static_cast<float>(v);
    }
}

template <typename DevT>
__device__ __forceinline__ DevT from_float_for_leaky(float f) {
    if constexpr (std::is_same_v<DevT, half>) {
        return __float2half_rn(f);
    } else if constexpr (std::is_same_v<DevT, __nv_bfloat16>) {
        return __float2bfloat16(f);
    } else {
        return static_cast<DevT>(f);
    }
}

template <class DevT>
__global__ void leakyrelu_kernel(
    DevT *__restrict__ out,
    const DevT *__restrict__ in,
    size_t n,
    float negative_slope,
    const size_t *__restrict__ shape,
    const size_t *__restrict__ div,
    const long long *__restrict__ in_stride,
    const long long *__restrict__ out_stride,
    int ndim) {

    size_t gid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t grid_stride = static_cast<size_t>(blockDim.x) * gridDim.x;

    for (size_t linear = gid; linear < n; linear += grid_stride) {
        unsigned long long rem = linear;
        long long in_off = 0;
        long long out_off = 0;
        for (int d = 0; d < ndim; ++d) {
            unsigned long long idx_d = 0;
            size_t divisor = div[d];
            if (divisor != 0) {
                idx_d = rem / divisor;
                rem = rem % divisor;
            } else {
                idx_d = 0;
            }
            if (in_stride[d] != 0) {
                in_off += static_cast<long long>(idx_d) * in_stride[d];
            }
            if (out_stride[d] != 0) {
                out_off += static_cast<long long>(idx_d) * out_stride[d];
            }
        }

        float v = to_float_for_leaky(in[static_cast<size_t>(in_off)]);
        float outv = v >= 0.0f ? v : v * negative_slope;
        out[static_cast<size_t>(out_off)] = from_float_for_leaky<DevT>(outv);
    }
}

#endif // __LEAKYRELU_CUDA_KERNEL_CUH__
