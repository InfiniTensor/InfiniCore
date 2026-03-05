#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <type_traits>

#include "../../../devices/nvidia/nvidia_kernel_common.cuh"

namespace op::prelu::cuda {

constexpr int kPreluMaxDims = 8;

struct TensorMeta {
    int ndim;
    size_t shape[kPreluMaxDims];
    ptrdiff_t strides[kPreluMaxDims]; // strides in elements
};

enum class WeightMode : int {
    SCALAR = 0,
    PER_CHANNEL = 1,
    ELEMENTWISE = 2,
};

template <typename T>
struct TypeTag {};

template <typename Tcompute>
__device__ __forceinline__ Tcompute to_compute(const half v) {
    return static_cast<Tcompute>(__half2float(v));
}

template <typename Tcompute>
__device__ __forceinline__ Tcompute to_compute(const cuda_bfloat16 v) {
    return static_cast<Tcompute>(__bfloat162float(v));
}

template <typename Tcompute, typename T>
__device__ __forceinline__ Tcompute to_compute(const T v) {
    return static_cast<Tcompute>(v);
}

__device__ __forceinline__ half from_compute(const float v, TypeTag<half>) {
    return __float2half_rn(v);
}

__device__ __forceinline__ cuda_bfloat16 from_compute(const float v, TypeTag<cuda_bfloat16>) {
    return __float2bfloat16_rn(v);
}

template <typename Tcompute, typename T>
__device__ __forceinline__ T from_compute(const Tcompute v, TypeTag<T>) {
    return static_cast<T>(v);
}

__device__ __forceinline__ size_t offset_from_flat(size_t flat, const TensorMeta &meta) {
    return device::nvidia::indexToOffset(
        flat,
        static_cast<size_t>(meta.ndim),
        meta.shape,
        meta.strides);
}

__device__ __forceinline__ size_t channel_from_flat(size_t flat, const TensorMeta &meta, int channel_axis) {
    size_t tmp = flat;
    size_t channel = 0;
    for (int d = meta.ndim - 1; d >= 0; --d) {
        const size_t coord = tmp % meta.shape[d];
        tmp /= meta.shape[d];
        if (d == channel_axis) {
            channel = coord;
        }
    }
    return channel;
}

template <typename T, typename Tcompute>
__global__ void prelu_kernel(
    T *output,
    const T *input,
    const T *weight,
    size_t n,
    TensorMeta out_meta,
    TensorMeta in_meta,
    int weight_mode,
    TensorMeta weight_meta,
    ptrdiff_t weight_stride0,
    int channel_axis) {

    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }

    const size_t in_off = offset_from_flat(idx, in_meta);
    const size_t out_off = offset_from_flat(idx, out_meta);

    const Tcompute x = to_compute<Tcompute>(input[in_off]);

    Tcompute w = 0;
    if (weight_mode == static_cast<int>(WeightMode::SCALAR)) {
        w = to_compute<Tcompute>(weight[0]);
    } else if (weight_mode == static_cast<int>(WeightMode::PER_CHANNEL)) {
        const size_t c = channel_from_flat(idx, in_meta, channel_axis);
        const size_t w_off = static_cast<size_t>(c * static_cast<size_t>(weight_stride0));
        w = to_compute<Tcompute>(weight[w_off]);
    } else { // ELEMENTWISE
        const size_t w_off = offset_from_flat(idx, weight_meta);
        w = to_compute<Tcompute>(weight[w_off]);
    }

    const Tcompute y = (x > Tcompute(0)) ? x : (w * x);
    output[out_off] = from_compute(y, TypeTag<T>{});
}

} // namespace op::prelu::cuda

