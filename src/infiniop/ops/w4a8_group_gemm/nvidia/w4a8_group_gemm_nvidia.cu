#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../../../tensor.h"
#include "w4a8_group_gemm_nvidia.cuh"

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {
constexpr size_t THREADS = 256;

__device__ __forceinline__ int unpack_signed_int4(uint8_t packed, bool high) {
    const int value = high ? (packed >> 4) : (packed & 0x0f);
    return value < 8 ? value : value - 16;
}

__device__ __forceinline__ int load_checkpoint_int4(
    const uint8_t *weight,
    size_t expert,
    size_t column,
    size_t inner,
    size_t n,
    size_t k,
    bool trans_weight) {
    if (trans_weight) {
        const size_t packed_inner = (inner / 32) * 16 + inner % 16;
        const bool high = (inner % 32) >= 16;
        const uint8_t packed = weight[(expert * n + column) * (k / 2) + packed_inner];
        return unpack_signed_int4(packed, high);
    }
    const size_t packed_column = (column / 32) * 16 + column % 16;
    const bool high = (column % 32) >= 16;
    const uint8_t packed = weight[(expert * k + inner) * (n / 2) + packed_column];
    return unpack_signed_int4(packed, high);
}

__device__ __forceinline__ uint32_t pack_signed_int4x4(
    uint32_t packed, bool high) {
    uint32_t unpacked = 0;
#pragma unroll
    for (size_t lane = 0; lane < 4; ++lane) {
        const uint8_t byte = static_cast<uint8_t>(packed >> (lane * 8));
        const int value = unpack_signed_int4(byte, high);
        unpacked |= static_cast<uint32_t>(static_cast<uint8_t>(value))
                 << (lane * 8);
    }
    return unpacked;
}

template <typename T>
__device__ __forceinline__ T from_float(float value);

template <>
__device__ __forceinline__ half from_float<half>(float value) {
    return __float2half_rn(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float value) {
    return __float2bfloat16(value);
}

template <typename T>
INFINIOP_CUDA_KERNEL w4a8GroupGemmKernel(
    T *__restrict__ out,
    const int8_t *__restrict__ input,
    const uint8_t *__restrict__ weight,
    const float *__restrict__ input_scale,
    const float *__restrict__ weight_scale,
    const int32_t *__restrict__ tokens_per_experts,
    const int32_t *__restrict__ sorted_token_ids,
    const T *__restrict__ bias,
    size_t m,
    size_t n,
    size_t k,
    size_t experts,
    bool trans_weight) {
    const size_t row = blockIdx.y;
    const size_t column = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || column >= n) {
        return;
    }

    __shared__ size_t selected_expert;
    if (threadIdx.x == 0) {
        size_t expert = 0;
        size_t expert_row_begin = 0;
        for (; expert < experts; ++expert) {
            const size_t expert_rows = static_cast<size_t>(tokens_per_experts[expert]);
            if (row < expert_row_begin + expert_rows) {
                break;
            }
            expert_row_begin += expert_rows;
        }
        selected_expert = expert;
    }
    __syncthreads();
    const size_t expert = selected_expert;
    if (expert == experts) {
        return;
    }

    int32_t acc = 0;
    if (trans_weight && (k % 32) == 0) {
        const auto *input_words = reinterpret_cast<const int32_t *>(input + row * k);
        const auto *weight_words = reinterpret_cast<const uint32_t *>(
            weight + (expert * n + column) * (k / 2));
        for (size_t block = 0; block < k / 32; ++block) {
#pragma unroll
            for (size_t lane = 0; lane < 4; ++lane) {
                const uint32_t packed = weight_words[block * 4 + lane];
                const int32_t low = static_cast<int32_t>(
                    pack_signed_int4x4(packed, false));
                const int32_t high = static_cast<int32_t>(
                    pack_signed_int4x4(packed, true));
                const size_t input_word = block * 8 + lane;
                acc = __dp4a(input_words[input_word], low, acc);
                acc = __dp4a(input_words[input_word + 4], high, acc);
            }
        }
    } else if ((k % 4) == 0) {
        const auto *input_words = reinterpret_cast<const int32_t *>(input + row * k);
        for (size_t inner = 0; inner < k; inner += 4) {
            uint32_t unpacked = 0;
            for (size_t lane = 0; lane < 4; ++lane) {
                const int value = load_checkpoint_int4(
                    weight, expert, column, inner + lane, n, k, trans_weight);
                unpacked |= static_cast<uint32_t>(static_cast<uint8_t>(value))
                         << (lane * 8);
            }
            acc = __dp4a(
                input_words[inner / 4], static_cast<int32_t>(unpacked), acc);
        }
    } else {
        for (size_t inner = 0; inner < k; ++inner) {
            const int value = load_checkpoint_int4(
                weight, expert, column, inner, n, k, trans_weight);
            acc += static_cast<int32_t>(input[row * k + inner]) * value;
        }
    }

    float result = static_cast<float>(acc) * 18.0f * input_scale[row]
                 * weight_scale[expert * n + column];
    if (bias != nullptr) {
        result += static_cast<float>(bias[expert * n + column]);
    }
    const size_t output_row = sorted_token_ids == nullptr
                                ? row
                                : static_cast<size_t>(sorted_token_ids[row]);
    if (output_row < m) {
        out[output_row * n + column] = from_float<T>(result);
    }
}
} // namespace

namespace op::w4a8_group_gemm::nvidia {

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t weight_desc,
    infiniopTensorDescriptor_t input_scale_desc,
    infiniopTensorDescriptor_t weight_scale_desc,
    infiniopTensorDescriptor_t tokens_per_experts_desc,
    infiniopTensorDescriptor_t sorted_token_ids_desc,
    infiniopTensorDescriptor_t bias_desc,
    bool trans_weight) {
    const auto out_shape = out_desc->shape();
    const auto input_shape = input_desc->shape();
    const auto weight_shape = weight_desc->shape();
    const auto input_scale_shape = input_scale_desc->shape();
    const auto weight_scale_shape = weight_scale_desc->shape();
    const auto tokens_shape = tokens_per_experts_desc->shape();
    CHECK_OR_RETURN(out_shape.size() == 2 && input_shape.size() == 2
                        && weight_shape.size() == 3 && input_scale_shape.size() == 2
                        && weight_scale_shape.size() == 3 && tokens_shape.size() == 1,
                    INFINI_STATUS_BAD_TENSOR_SHAPE);
    const size_t m = input_shape[0];
    const size_t experts = weight_shape[0];
    const size_t k = trans_weight ? weight_shape[2] * 2 : weight_shape[1];
    const size_t n = trans_weight ? weight_shape[1] : weight_shape[2] * 2;
    CHECK_OR_RETURN((k % 2) == 0 && (n % 2) == 0
                        && (trans_weight ? (k % 32) == 0 : (n % 32) == 0)
                        && out_shape[0] == m && out_shape[1] == n && input_shape[1] == k
                        && input_scale_shape[0] == m && input_scale_shape[1] == 1
                        && weight_scale_shape[0] == experts
                        && weight_scale_shape[1] == n && weight_scale_shape[2] == 1
                        && tokens_shape[0] == experts
                        && (sorted_token_ids_desc == nullptr
                            || (sorted_token_ids_desc->shape().size() == 1
                                && sorted_token_ids_desc->shape()[0] == m))
                        && (bias_desc == nullptr
                            || (bias_desc->shape().size() == 2
                                && bias_desc->shape()[0] == experts
                                && bias_desc->shape()[1] == n)),
                    INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(out_desc->isContiguous() && input_desc->isContiguous()
                        && weight_desc->isContiguous() && input_scale_desc->isContiguous()
                        && weight_scale_desc->isContiguous() && tokens_per_experts_desc->isContiguous()
                        && (sorted_token_ids_desc == nullptr || sorted_token_ids_desc->isContiguous())
                        && (bias_desc == nullptr || bias_desc->isContiguous()),
                    INFINI_STATUS_BAD_TENSOR_STRIDES);
    CHECK_OR_RETURN(input_desc->dtype() == INFINI_DTYPE_I8
                        && weight_desc->dtype() == INFINI_DTYPE_I8
                        && input_scale_desc->dtype() == INFINI_DTYPE_F32
                        && weight_scale_desc->dtype() == INFINI_DTYPE_F32
                        && tokens_per_experts_desc->dtype() == INFINI_DTYPE_I32
                        && (out_desc->dtype() == INFINI_DTYPE_F16
                            || out_desc->dtype() == INFINI_DTYPE_BF16)
                        && (sorted_token_ids_desc == nullptr
                            || sorted_token_ids_desc->dtype() == INFINI_DTYPE_I32)
                        && (bias_desc == nullptr || bias_desc->dtype() == out_desc->dtype()),
                    INFINI_STATUS_BAD_TENSOR_DTYPE);
    *desc_ptr = new Descriptor(m, n, k, experts, out_desc->dtype(), trans_weight,
                               sorted_token_ids_desc != nullptr, bias_desc != nullptr,
                               handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *out, const void *input, const void *weight, const void *input_scale,
    const void *weight_scale, const void *tokens_per_experts,
    const void *sorted_token_ids, const void *bias, void *stream) const {
    const dim3 grid(static_cast<unsigned int>((_n + THREADS - 1) / THREADS),
                    static_cast<unsigned int>(_m));
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (_out_dtype == INFINI_DTYPE_F16) {
        w4a8GroupGemmKernel<half><<<grid, THREADS, 0, cuda_stream>>>(
            static_cast<half *>(out), static_cast<const int8_t *>(input),
            static_cast<const uint8_t *>(weight), static_cast<const float *>(input_scale),
            static_cast<const float *>(weight_scale), static_cast<const int32_t *>(tokens_per_experts),
            static_cast<const int32_t *>(sorted_token_ids), static_cast<const half *>(bias),
            _m, _n, _k, _experts, _trans_weight);
    } else {
        w4a8GroupGemmKernel<__nv_bfloat16><<<grid, THREADS, 0, cuda_stream>>>(
            static_cast<__nv_bfloat16 *>(out), static_cast<const int8_t *>(input),
            static_cast<const uint8_t *>(weight), static_cast<const float *>(input_scale),
            static_cast<const float *>(weight_scale), static_cast<const int32_t *>(tokens_per_experts),
            static_cast<const int32_t *>(sorted_token_ids), static_cast<const __nv_bfloat16 *>(bias),
            _m, _n, _k, _experts, _trans_weight);
    }
    return cudaGetLastError() == cudaSuccess ? INFINI_STATUS_SUCCESS
                                             : INFINI_STATUS_INTERNAL_ERROR;
}

} // namespace op::w4a8_group_gemm::nvidia
