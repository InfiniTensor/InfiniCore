#ifndef __CAST_CUDA_KERNEL_CUH__
#define __CAST_CUDA_KERNEL_CUH__

#include <cuda_fp16.h>
#include <stdint.h>
#include <type_traits>

template <typename Tout, typename Tin>
__device__ __forceinline__ Tout device_cast(const Tin &v) {
    if constexpr (std::is_same_v<Tout, half>) {
        float f;
        if constexpr (std::is_same_v<Tin, half>) {
            f = __half2float(v);
        } else {
            f = static_cast<float>(v);
        }
        return __float2half_rn(f);
    } else if constexpr (std::is_same_v<Tout, float>) {
        if constexpr (std::is_same_v<Tin, half>) {
            return __half2float(v);
        } else {
            return static_cast<float>(v);
        }
    } else if constexpr (std::is_same_v<Tout, double>) {
        if constexpr (std::is_same_v<Tin, half>) {
            return static_cast<double>(__half2float(v));
        } else {
            return static_cast<double>(v);
        }
    } else { // integer outputs
        // convert via double/float then to integer (truncate)
        if constexpr (std::is_same_v<Tin, half>) {
            float f = __half2float(v);
            return static_cast<Tout>(f);
        } else {
            return static_cast<Tout>(v);
        }
    }
}

template <class ToutDev, class TinDev>
__global__ void cast_kernel(
    ToutDev *__restrict__ out,
    const TinDev *__restrict__ in,
    size_t n,
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
        out[static_cast<size_t>(out_off)] = device_cast<ToutDev, TinDev>(in[static_cast<size_t>(in_off)]);
    }
}

#endif // __CAST_CUDA_KERNEL_CUH__
