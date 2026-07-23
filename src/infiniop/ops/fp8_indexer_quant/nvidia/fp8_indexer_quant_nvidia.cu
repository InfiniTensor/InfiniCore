#include "../../../../utils.h"
#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../../../tensor.h"
#include "fp8_indexer_quant_nvidia.cuh"

#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

namespace {
template <typename T>
__device__ __forceinline__ float to_float(T value);

template <>
__device__ __forceinline__ float to_float<half>(half value) {
    return __half2float(value);
}

template <>
__device__ __forceinline__ float to_float<cuda_bfloat16>(cuda_bfloat16 value) {
    return __bfloat162float(value);
}

template <typename T>
__device__ __forceinline__ T from_float(float value);

template <>
__device__ __forceinline__ half from_float<half>(float value) {
    return __float2half_rn(value);
}

template <>
__device__ __forceinline__ cuda_bfloat16 from_float<cuda_bfloat16>(float value) {
    return __float2bfloat16_rn(value);
}

__device__ __forceinline__ float vendor_fp8_quotient(float value, float scale) {
    const float reciprocal = __frcp_rn(scale);
    const float quotient = value * reciprocal;
    const float residual = fmaf(-scale, quotient, value);
    return fmaf(residual, reciprocal, quotient);
}

__device__ __forceinline__ uint8_t vendor_fp8_e4m3(float value) {
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

template <typename T>
INFINIOP_CUDA_KERNEL fp8IndexerQuantKernel(
    cuda_fp8_e4m3 *__restrict__ q_fp8,
    float *__restrict__ weights_fp32,
    const T *__restrict__ q,
    const T *__restrict__ weights,
    size_t head_dim) {
    const size_t group = blockIdx.x;
    const size_t column = threadIdx.x;
    const float value = column < head_dim
                          ? to_float(q[group * head_dim + column])
                          : 0.0f;

    extern __shared__ float reduction[];
    reduction[column] = fabsf(value);
    __syncthreads();
    for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (column < stride) {
            reduction[column] = fmaxf(reduction[column], reduction[column + stride]);
        }
        __syncthreads();
    }

    if (column == 0) {
        const float abs_max = fmaxf(reduction[0], 1.0e-4f);
        const float scale_raw = abs_max / 448.0f;
        reduction[0] = exp2f(ceilf(log2f(scale_raw)));
        weights_fp32[group] = to_float(weights[group]) * reduction[0];
    }
    __syncthreads();

    if (column < head_dim) {
        const float quantized = fminf(448.0f, fmaxf(-448.0f, value / reduction[0]));
        q_fp8[group * head_dim + column] = cuda_fp8_e4m3(quantized);
    }
}

template <typename T>
INFINIOP_CUDA_KERNEL fusedFp8IndexerKernel(
    cuda_fp8_e4m3 *__restrict__ q_fp8,
    float *__restrict__ weights_fp32,
    uint8_t *__restrict__ k_cache,
    const T *__restrict__ q_raw,
    const T *__restrict__ k_weights,
    const T *__restrict__ norm_weight,
    const T *__restrict__ norm_bias,
    const int64_t *__restrict__ positions,
    const T *__restrict__ cos_sin_cache,
    const int64_t *__restrict__ slot_mapping,
    size_t num_heads,
    size_t head_dim,
    size_t rope_dim,
    size_t num_cache_blocks,
    size_t block_size,
    size_t cache_stride,
    size_t max_positions,
    float eps,
    float weights_scale) {
    const size_t work_per_token = num_heads + 1;
    const size_t token = blockIdx.x / work_per_token;
    const size_t work = blockIdx.x % work_per_token;
    const size_t column = threadIdx.x;
    const int64_t position_raw = positions[token];
    const size_t position = position_raw >= 0
                                 && static_cast<size_t>(position_raw) < max_positions
                              ? static_cast<size_t>(position_raw)
                              : 0;
    const size_t half_rope_dim = rope_dim / 2;

    extern __shared__ float shared[];
    float *values = shared;
    float *scratch = shared + head_dim;

    if (work < num_heads) {
        const size_t group = token * num_heads + work;
        const size_t q_base = group * head_dim;
        float value = to_float(q_raw[q_base + column]);
        if (column < rope_dim) {
            const size_t pair = column & ~size_t(1);
            const float x0 = to_float(q_raw[q_base + pair]);
            const float x1 = to_float(q_raw[q_base + pair + 1]);
            const size_t angle = column / 2;
            const size_t cache_base = position * rope_dim;
            const float cosine = to_float(cos_sin_cache[cache_base + angle]);
            const float sine = to_float(
                cos_sin_cache[cache_base + half_rope_dim + angle]);
            value = (column & 1) != 0
                      ? x0 * sine + x1 * cosine
                      : x0 * cosine - x1 * sine;
        }
        value = to_float(from_float<T>(value));
        scratch[column] = fabsf(value);
        __syncthreads();
        for (size_t stride = head_dim / 2; stride > 0; stride >>= 1) {
            if (column < stride) {
                scratch[column] = fmaxf(scratch[column], scratch[column + stride]);
            }
            __syncthreads();
        }
        if (column == 0) {
            const float abs_max = fmaxf(scratch[0], 1.0e-4f);
            const float scale_raw = abs_max / 448.0f;
            scratch[0] = exp2f(ceilf(log2f(scale_raw)));
            const T scaled_weight = from_float<T>(
                to_float(k_weights[token * (head_dim + num_heads)
                                   + head_dim + work])
                * weights_scale);
            weights_fp32[group] = to_float(scaled_weight) * scratch[0];
        }
        __syncthreads();
        const float quantized = fminf(
            448.0f, fmaxf(-448.0f, value / scratch[0]));
        q_fp8[q_base + column] = cuda_fp8_e4m3(quantized);
        return;
    }

    const int64_t slot_raw = slot_mapping[token];
    if (slot_raw < 0
        || static_cast<size_t>(slot_raw) >= num_cache_blocks * block_size) {
        return;
    }
    const size_t kw_base = token * (head_dim + num_heads);
    const size_t half_head_dim = head_dim / 2;
    const bool active_lane = column < half_head_dim;
    const size_t column0 = column * 2;
    const size_t column1 = column0 + 1;
    float input0 = 0.0f;
    float input1 = 0.0f;
    if (active_lane) {
        input0 = to_float(k_weights[kw_base + column0]);
        input1 = to_float(k_weights[kw_base + column1]);
        // Match the vendor fused kernel: one 64-lane wave owns an adjacent
        // BF16 pair and accumulates x0^2 with a fused multiply-add.
        values[column] = input0 + input1;
        scratch[column] = fmaf(input0, input0, input1 * input1);
    }
    __syncthreads();
    for (size_t stride = half_head_dim / 2; stride > 0; stride >>= 1) {
        if (column < stride) {
            values[column] += values[column + stride];
            scratch[column] += scratch[column + stride];
        }
        __syncthreads();
    }
    const float inv_head_dim = 1.0f / static_cast<float>(head_dim);
    const float mean = values[0] * inv_head_dim;
    const float variance = fmaf(scratch[0], inv_head_dim, -mean * mean);
    const float inv_std = rsqrtf(variance + eps);
    if (active_lane) {
        const float normalized0 = fmaf(
            (input0 - mean) * inv_std,
            to_float(norm_weight[column0]),
            to_float(norm_bias[column0]));
        const float normalized1 = fmaf(
            (input1 - mean) * inv_std,
            to_float(norm_weight[column1]),
            to_float(norm_bias[column1]));
        values[column0] = normalized0;
        values[column1] = normalized1;
    }
    __syncthreads();

    float value0 = 0.0f;
    float value1 = 0.0f;
    if (active_lane) {
        value0 = values[column0];
        value1 = values[column1];
    }
    if (active_lane && column0 < rope_dim) {
        const float x0 = value0;
        const float x1 = value1;
        const size_t angle = column;
        const size_t cache_base = position * rope_dim;
        const float cosine = to_float(cos_sin_cache[cache_base + angle]);
        const float sine = to_float(
            cos_sin_cache[cache_base + half_rope_dim + angle]);
        const float x1_sine = x1 * sine;
        const float x0_sine = x0 * sine;
        value0 = fmaf(x0, cosine, -x1_sine);
        value1 = fmaf(x1, cosine, x0_sine);
    }
    if (active_lane) {
        value0 = to_float(from_float<T>(value0));
        value1 = to_float(from_float<T>(value1));
    }
    if (active_lane) {
        scratch[column] = fmaxf(fabsf(value0), fabsf(value1));
    }
    __syncthreads();
    for (size_t stride = half_head_dim / 2; stride > 0; stride >>= 1) {
        if (column < stride) {
            scratch[column] = fmaxf(scratch[column], scratch[column + stride]);
        }
        __syncthreads();
    }
    if (column == 0) {
        const float abs_max = fmaxf(scratch[0], 1.0e-4f);
        const float scale_raw = abs_max / 448.0f;
        scratch[0] = exp2f(ceilf(log2f(scale_raw)));
    }
    __syncthreads();

    const size_t slot = static_cast<size_t>(slot_raw);
    const size_t cache_block = slot / block_size;
    const size_t cache_offset = slot % block_size;
    uint8_t *cache_base = k_cache
                        + cache_block * block_size * cache_stride;
    if (active_lane) {
        uint8_t *cache_values = cache_base + cache_offset * head_dim;
        const float quantized0 = fminf(
            448.0f, fmaxf(-448.0f, vendor_fp8_quotient(value0, scratch[0])));
        const float quantized1 = fminf(
            448.0f, fmaxf(-448.0f, vendor_fp8_quotient(value1, scratch[0])));
        cache_values[column0] = vendor_fp8_e4m3(quantized0);
        cache_values[column1] = vendor_fp8_e4m3(quantized1);
    }
    if (column == 0) {
        *reinterpret_cast<float *>(
            cache_base + block_size * head_dim
            + cache_offset * sizeof(float))
            = scratch[0];
    }
}
} // namespace

namespace op::fp8_indexer_quant::nvidia {

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t q_fp8_desc,
    infiniopTensorDescriptor_t weights_fp32_desc,
    infiniopTensorDescriptor_t q_desc,
    infiniopTensorDescriptor_t weights_desc) {
    const auto q_shape = q_desc->shape();
    const auto weights_shape = weights_desc->shape();
    CHECK_OR_RETURN(q_shape.size() == 3 && weights_shape.size() == 2,
                    INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(q_fp8_desc->shape() == q_shape
                        && weights_fp32_desc->shape() == weights_shape
                        && weights_shape[0] == q_shape[0]
                        && weights_shape[1] == q_shape[1],
                    INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(q_desc->isContiguous() && weights_desc->isContiguous()
                        && q_fp8_desc->isContiguous()
                        && weights_fp32_desc->isContiguous(),
                    INFINI_STATUS_BAD_TENSOR_STRIDES);
    const auto input_dtype = q_desc->dtype();
    CHECK_OR_RETURN(input_dtype == INFINI_DTYPE_F16
                        || input_dtype == INFINI_DTYPE_BF16,
                    INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(weights_desc->dtype() == input_dtype
                        && q_fp8_desc->dtype() == INFINI_DTYPE_F8
                        && weights_fp32_desc->dtype() == INFINI_DTYPE_F32,
                    INFINI_STATUS_BAD_TENSOR_DTYPE);
    const size_t head_dim = q_shape[2];
    CHECK_OR_RETURN(head_dim > 0 && head_dim <= 1024,
                    INFINI_STATUS_BAD_TENSOR_SHAPE);
    size_t threads = 1;
    while (threads < head_dim) {
        threads <<= 1;
    }
    *desc_ptr = new Descriptor(
        q_shape[0] * q_shape[1], head_dim, threads, input_dtype,
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *q_fp8,
    void *weights_fp32,
    const void *q,
    const void *weights,
    void *stream) const {
    const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    const size_t smem = _threads * sizeof(float);
    if (_input_dtype == INFINI_DTYPE_F16) {
        fp8IndexerQuantKernel<half><<<_num_groups, _threads, smem, cuda_stream>>>(
            reinterpret_cast<cuda_fp8_e4m3 *>(q_fp8),
            reinterpret_cast<float *>(weights_fp32),
            reinterpret_cast<const half *>(q),
            reinterpret_cast<const half *>(weights),
            _head_dim);
    } else {
        fp8IndexerQuantKernel<cuda_bfloat16><<<_num_groups, _threads, smem, cuda_stream>>>(
            reinterpret_cast<cuda_fp8_e4m3 *>(q_fp8),
            reinterpret_cast<float *>(weights_fp32),
            reinterpret_cast<const cuda_bfloat16 *>(q),
            reinterpret_cast<const cuda_bfloat16 *>(weights),
            _head_dim);
    }
    return cudaGetLastError() == cudaSuccess
             ? INFINI_STATUS_SUCCESS
             : INFINI_STATUS_INTERNAL_ERROR;
}

infiniStatus_t FusedDescriptor::create(
    infiniopHandle_t handle,
    FusedDescriptor **desc_ptr,
    infiniopTensorDescriptor_t q_fp8_desc,
    infiniopTensorDescriptor_t weights_fp32_desc,
    infiniopTensorDescriptor_t k_cache_desc,
    infiniopTensorDescriptor_t q_raw_desc,
    infiniopTensorDescriptor_t k_weights_desc,
    infiniopTensorDescriptor_t norm_weight_desc,
    infiniopTensorDescriptor_t norm_bias_desc,
    infiniopTensorDescriptor_t positions_desc,
    infiniopTensorDescriptor_t cos_sin_cache_desc,
    infiniopTensorDescriptor_t slot_mapping_desc,
    uint64_t rope_dim,
    double eps,
    double weights_scale) {
    const auto q_shape = q_raw_desc->shape();
    const auto weights_shape = weights_fp32_desc->shape();
    const auto kw_shape = k_weights_desc->shape();
    const auto cache_shape = k_cache_desc->shape();
    const auto cos_sin_shape = cos_sin_cache_desc->shape();
    CHECK_OR_RETURN(q_shape.size() == 3 && weights_shape.size() == 2
                        && kw_shape.size() == 2 && cache_shape.size() == 3
                        && norm_weight_desc->shape().size() == 1
                        && norm_bias_desc->shape().size() == 1
                        && positions_desc->shape().size() == 1
                        && cos_sin_shape.size() == 2
                        && slot_mapping_desc->shape().size() == 1,
                    INFINI_STATUS_BAD_TENSOR_SHAPE);
    const size_t num_tokens = q_shape[0];
    const size_t num_heads = q_shape[1];
    const size_t head_dim = q_shape[2];
    CHECK_OR_RETURN(head_dim == 128 && rope_dim > 0 && rope_dim <= head_dim
                        && rope_dim % 2 == 0
                        && q_fp8_desc->shape() == q_shape
                        && weights_shape[0] == num_tokens
                        && weights_shape[1] == num_heads
                        && kw_shape[0] == num_tokens
                        && kw_shape[1] == head_dim + num_heads
                        && norm_weight_desc->shape()[0] == head_dim
                        && norm_bias_desc->shape()[0] == head_dim
                        && positions_desc->shape()[0] == num_tokens
                        && slot_mapping_desc->shape()[0] == num_tokens
                        && cos_sin_shape[1] == rope_dim
                        && cache_shape[1] > 0
                        && cache_shape[2] == head_dim + sizeof(float),
                    INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(q_fp8_desc->isContiguous()
                        && weights_fp32_desc->isContiguous()
                        && k_cache_desc->isContiguous()
                        && q_raw_desc->isContiguous()
                        && k_weights_desc->isContiguous()
                        && norm_weight_desc->isContiguous()
                        && norm_bias_desc->isContiguous()
                        && positions_desc->isContiguous()
                        && cos_sin_cache_desc->isContiguous()
                        && slot_mapping_desc->isContiguous(),
                    INFINI_STATUS_BAD_TENSOR_STRIDES);
    const auto input_dtype = q_raw_desc->dtype();
    CHECK_OR_RETURN(input_dtype == INFINI_DTYPE_F16
                        || input_dtype == INFINI_DTYPE_BF16,
                    INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(k_weights_desc->dtype() == input_dtype
                        && norm_weight_desc->dtype() == input_dtype
                        && norm_bias_desc->dtype() == input_dtype
                        && cos_sin_cache_desc->dtype() == input_dtype
                        && q_fp8_desc->dtype() == INFINI_DTYPE_F8
                        && weights_fp32_desc->dtype() == INFINI_DTYPE_F32
                        && k_cache_desc->dtype() == INFINI_DTYPE_U8
                        && positions_desc->dtype() == INFINI_DTYPE_I64
                        && slot_mapping_desc->dtype() == INFINI_DTYPE_I64,
                    INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(eps > 0.0 && weights_scale > 0.0,
                    INFINI_STATUS_BAD_PARAM);
    *desc_ptr = new FusedDescriptor(
        num_tokens, num_heads, head_dim, rope_dim,
        cache_shape[0], cache_shape[1], cache_shape[2], cos_sin_shape[0],
        input_dtype, eps, weights_scale, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t FusedDescriptor::calculate(
    void *q_fp8, void *weights_fp32, void *k_cache,
    const void *q_raw, const void *k_weights,
    const void *norm_weight, const void *norm_bias,
    const void *positions, const void *cos_sin_cache,
    const void *slot_mapping, void *stream) const {
    const size_t blocks = _num_tokens * (_num_heads + 1);
    const size_t smem = 2 * _head_dim * sizeof(float);
#define LAUNCH_FUSED(T)                                                   \
    fusedFp8IndexerKernel<T><<<blocks, _head_dim, smem,                   \
                               reinterpret_cast<cudaStream_t>(stream)>>>( \
        reinterpret_cast<cuda_fp8_e4m3 *>(q_fp8),                         \
        reinterpret_cast<float *>(weights_fp32),                          \
        reinterpret_cast<uint8_t *>(k_cache),                             \
        reinterpret_cast<const T *>(q_raw),                               \
        reinterpret_cast<const T *>(k_weights),                           \
        reinterpret_cast<const T *>(norm_weight),                         \
        reinterpret_cast<const T *>(norm_bias),                           \
        reinterpret_cast<const int64_t *>(positions),                     \
        reinterpret_cast<const T *>(cos_sin_cache),                       \
        reinterpret_cast<const int64_t *>(slot_mapping),                  \
        _num_heads, _head_dim, _rope_dim, _num_cache_blocks, _block_size, \
        _cache_stride, _max_positions, _eps, _weights_scale)
    if (_input_dtype == INFINI_DTYPE_F16) {
        LAUNCH_FUSED(half);
    } else {
        LAUNCH_FUSED(cuda_bfloat16);
    }
#undef LAUNCH_FUSED
    return cudaGetLastError() == cudaSuccess
             ? INFINI_STATUS_SUCCESS
             : INFINI_STATUS_INTERNAL_ERROR;
}

} // namespace op::fp8_indexer_quant::nvidia
