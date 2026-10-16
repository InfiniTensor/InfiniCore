#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../../../tensor.h"
#include "infiniop/ops/moe_weighted_sum.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {
constexpr size_t THREADS = 256;

template <typename T>
__device__ float to_float(T value);
template <>
__device__ float to_float(half value) { return __half2float(value); }
template <>
__device__ float to_float(__nv_bfloat16 value) { return __bfloat162float(value); }

template <typename T>
__device__ T from_float(float value);
template <>
__device__ half from_float(float value) { return __float2half_rn(value); }
template <>
__device__ __nv_bfloat16 from_float(float value) { return __float2bfloat16(value); }

template <typename T>
INFINIOP_CUDA_KERNEL moeWeightedSumKernel(
    T *output,
    const T *input,
    const float *topk_weights,
    const T *residual,
    size_t tokens,
    size_t topk,
    size_t hidden,
    float routed_scale,
    float residual_scale) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= tokens * hidden) {
        return;
    }
    const size_t token = index / hidden;
    const size_t column = index % hidden;
    float value = 0.0f;
    for (size_t route = 0; route < topk; ++route) {
        float weight = topk_weights == nullptr
                         ? 1.0f
                         : topk_weights[token * topk + route];
        value += to_float(input[(token * topk + route) * hidden + column])
               * weight;
    }
    value *= routed_scale;
    if (residual != nullptr) {
        value += to_float(residual[index]) * residual_scale;
    }
    output[index] = from_float<T>(value);
}
} // namespace

__INFINI_C infiniStatus_t infiniopMoeWeightedSum(
    infiniopHandle_t handle,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t topk_weights_desc,
    infiniopTensorDescriptor_t residual_desc,
    void *output,
    const void *input,
    const void *topk_weights,
    const void *residual,
    double routed_scale,
    double residual_scale,
    void *stream) {
    (void)handle;
    const auto output_shape = output_desc->shape();
    const auto input_shape = input_desc->shape();
    CHECK_OR_RETURN(output_shape.size() == 2 && input_shape.size() == 3,
                    INFINI_STATUS_BAD_TENSOR_SHAPE);
    const size_t tokens = input_shape[0], topk = input_shape[1], hidden = input_shape[2];
    CHECK_OR_RETURN(output_shape[0] == tokens && output_shape[1] == hidden
                        && (topk_weights_desc == nullptr
                            || (topk_weights_desc->shape().size() == 2
                                && topk_weights_desc->shape()[0] == tokens
                                && topk_weights_desc->shape()[1] == topk))
                        && (residual_desc == nullptr || residual_desc->shape() == output_shape),
                    INFINI_STATUS_BAD_TENSOR_SHAPE);
    const auto dtype = input_desc->dtype();
    CHECK_OR_RETURN((dtype == INFINI_DTYPE_F16 || dtype == INFINI_DTYPE_BF16)
                        && output_desc->dtype() == dtype
                        && (topk_weights_desc == nullptr
                            || topk_weights_desc->dtype() == INFINI_DTYPE_F32)
                        && (residual_desc == nullptr || residual_desc->dtype() == dtype),
                    INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(output_desc->isContiguous() && input_desc->isContiguous()
                        && (topk_weights_desc == nullptr || topk_weights_desc->isContiguous())
                        && (residual_desc == nullptr || residual_desc->isContiguous()),
                    INFINI_STATUS_BAD_TENSOR_STRIDES);
    const size_t blocks = (tokens * hidden + THREADS - 1) / THREADS;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (dtype == INFINI_DTYPE_F16) {
        moeWeightedSumKernel<<<blocks, THREADS, 0, cuda_stream>>>(
            static_cast<half *>(output), static_cast<const half *>(input),
            static_cast<const float *>(topk_weights), static_cast<const half *>(residual),
            tokens, topk, hidden, routed_scale, residual_scale);
    } else {
        moeWeightedSumKernel<<<blocks, THREADS, 0, cuda_stream>>>(
            static_cast<__nv_bfloat16 *>(output), static_cast<const __nv_bfloat16 *>(input),
            static_cast<const float *>(topk_weights), static_cast<const __nv_bfloat16 *>(residual),
            tokens, topk, hidden, routed_scale, residual_scale);
    }
    return cudaGetLastError() == cudaSuccess ? INFINI_STATUS_SUCCESS
                                             : INFINI_STATUS_INTERNAL_ERROR;
}
