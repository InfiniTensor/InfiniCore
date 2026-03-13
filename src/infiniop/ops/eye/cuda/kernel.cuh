#ifndef __EYE_CUDA_H__
#define __EYE_CUDA_H__

#include "../../../devices/nvidia/nvidia_kernel_common.cuh"

namespace op::eye::cuda {

template <typename T>
__device__ __forceinline__ void eye_kernel_element(T *out, size_t idx, size_t n, size_t m) {
    size_t row = idx / m;
    size_t col = idx % m;
    out[idx] = (row == col) ? static_cast<T>(1) : static_cast<T>(0);
}

template <>
__device__ __forceinline__ void eye_kernel_element<half>(half *out, size_t idx, size_t n, size_t m) {
    size_t row = idx / m;
    size_t col = idx % m;
    out[idx] = (row == col) ? __float2half_rn(1.0f) : __float2half_rn(0.0f);
}

template <>
__device__ __forceinline__ void eye_kernel_element<cuda_bfloat16>(cuda_bfloat16 *out, size_t idx,
                                                                 size_t n, size_t m) {
    size_t row = idx / m;
    size_t col = idx % m;
    out[idx] = (row == col) ? __float2bfloat16_rn(1.0f) : __float2bfloat16_rn(0.0f);
}

template <typename T>
__global__ void eyeKernel(T *out, size_t n, size_t m) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * m) {
        eye_kernel_element<T>(out, idx, n, m);
    }
}

} // namespace op::eye::cuda

#endif
