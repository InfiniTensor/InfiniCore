/*
 * Portions of the CUDA kernels in this file are adapted from SGLang:
 * /sgl-kernel/csrc/moe/moe_fused_gate.cu
 *
 * Copyright 2025 SGLang Team. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0.
 */

#include "moe_fused_gate_nvidia.cuh"

#include "../../../devices/nvidia/nvidia_common.cuh"

#include <cfloat>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <memory>
#include <type_traits>
#include <utility>

namespace op::moe_fused_gate::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

namespace {

constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_CTA = 6;
constexpr int MAX_VPT = 32;

template <typename T>
__device__ float convert_to_float(T x) {
    if constexpr (std::is_same_v<T, half>) {
        return __half2float(x);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return __bfloat162float(x);
    } else if constexpr (std::is_same_v<T, float>) {
        return x;
    } else {
        return static_cast<float>(x);
    }
}

template <int VPT_, int NUM_EXPERTS_, int THREADS_PER_ROW_, int ROWS_PER_WARP_, int ROWS_PER_CTA_, int WARPS_PER_CTA_>
struct KernelParams {
    static constexpr int VPT = VPT_;
    static constexpr int NUM_EXPERTS = NUM_EXPERTS_;
    static constexpr int THREADS_PER_ROW = THREADS_PER_ROW_;
    static constexpr int ROWS_PER_WARP = ROWS_PER_WARP_;
    static constexpr int ROWS_PER_CTA = ROWS_PER_CTA_;
    static constexpr int WARPS_PER_CTA = WARPS_PER_CTA_;
};

struct KernelParamsDynamic {
    int VPT;
    int NUM_EXPERTS;
    int THREADS_PER_ROW;
    int ROWS_PER_WARP;
    int ROWS_PER_CTA;
    int WARPS_PER_CTA;
};

template <typename T, typename Params>
__device__ void moe_fused_gate_impl(
    const T *__restrict__ input,
    const float *__restrict__ bias,
    float *__restrict__ output,
    int32_t *__restrict__ indices,
    int64_t num_rows,
    int64_t topk_group,
    int64_t topk,
    int64_t num_fused_shared_experts,
    float routed_scaling_factor,
    bool apply_routed_scaling_factor_on_output,
    Params params) {
    const int tidx = threadIdx.x;
    const int64_t thread_row = blockIdx.x * params.ROWS_PER_CTA + threadIdx.y * params.ROWS_PER_WARP + tidx / params.THREADS_PER_ROW;
    if (thread_row >= num_rows) {
        return;
    }

    const int64_t routed_topk = topk - num_fused_shared_experts;
    const T *thread_row_ptr = input + thread_row * params.NUM_EXPERTS;
    const int thread_group_idx = tidx % params.THREADS_PER_ROW;
    const int first_elt_read_by_thread = thread_group_idx * params.VPT;

    float row_chunk[MAX_VPT];
    float bias_chunk[MAX_VPT];
#pragma unroll
    for (int ii = 0; ii < MAX_VPT; ++ii) {
        if (ii < params.VPT) {
            const int expert = first_elt_read_by_thread + ii;
            const float sigmoid = 1.0f / (1.0f + expf(-convert_to_float(thread_row_ptr[expert])));
            row_chunk[ii] = sigmoid;
            bias_chunk[ii] = sigmoid + bias[expert];
        }
    }

    __syncthreads();

    for (int k_idx = 0; k_idx < params.THREADS_PER_ROW - topk_group; ++k_idx) {
        int expert = first_elt_read_by_thread;
        float max_val = -FLT_MAX;
        float max_val_second = -FLT_MAX;
#pragma unroll
        for (int ii = 0; ii < MAX_VPT; ++ii) {
            if (ii < params.VPT) {
                const float val = bias_chunk[ii];
                if (val > max_val) {
                    max_val_second = max_val;
                    max_val = val;
                } else if (val > max_val_second) {
                    max_val_second = val;
                }
            }
        }
        float max_sum = max_val + max_val_second;

#pragma unroll
        for (int mask = params.THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
            const float other_max_sum = __shfl_xor_sync(0xffffffff, max_sum, mask, params.THREADS_PER_ROW);
            const int other_expert = __shfl_xor_sync(0xffffffff, expert, mask, params.THREADS_PER_ROW);
            if (max_sum > other_max_sum || (other_max_sum == max_sum && other_expert > expert)) {
                max_sum = other_max_sum;
                expert = other_expert;
            }
        }

        const int thread_to_clear_in_group = expert / params.VPT;
        if (thread_group_idx == thread_to_clear_in_group) {
#pragma unroll
            for (int ii = 0; ii < MAX_VPT; ++ii) {
                if (ii < params.VPT) {
                    bias_chunk[ii] = FLT_MAX;
                }
            }
        }
    }

    __syncthreads();

    float output_sum = 0.0f;
    for (int k_idx = 0; k_idx < routed_topk; ++k_idx) {
        float max_val = bias_chunk[0];
        int expert = first_elt_read_by_thread;
        if (max_val != FLT_MAX) {
#pragma unroll
            for (int ii = 1; ii < MAX_VPT; ++ii) {
                if (ii < params.VPT) {
                    const float val = bias_chunk[ii];
                    if (val > max_val) {
                        max_val = val;
                        expert = first_elt_read_by_thread + ii;
                    }
                }
            }
        } else {
            max_val = -FLT_MAX;
        }

#pragma unroll
        for (int mask = params.THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
            const float other_max = __shfl_xor_sync(0xffffffff, max_val, mask, params.THREADS_PER_ROW);
            const int other_expert = __shfl_xor_sync(0xffffffff, expert, mask, params.THREADS_PER_ROW);
            if (other_max > max_val || (other_max == max_val && other_expert < expert)) {
                max_val = other_max;
                expert = other_expert;
            }
        }

        const int thread_to_clear_in_group = expert / params.VPT;
        const int64_t idx = topk * thread_row + k_idx;
        if (thread_group_idx == thread_to_clear_in_group) {
            const int expert_to_clear_in_thread = expert % params.VPT;
            bias_chunk[expert_to_clear_in_thread] = -FLT_MAX;
            output[idx] = row_chunk[expert_to_clear_in_thread];
            indices[idx] = static_cast<int32_t>(expert);
        }

        if (thread_group_idx == 0) {
            output_sum += output[idx];
        }
        __syncthreads();
    }

    if (thread_group_idx == 0 && num_fused_shared_experts > 0) {
        int64_t idx = topk * thread_row + routed_topk;
        for (int i = 0; i < num_fused_shared_experts; ++i) {
            indices[idx + i] = static_cast<int32_t>(params.NUM_EXPERTS + i);
            output[idx + i] = output_sum / routed_scaling_factor;
        }
    }
    __syncthreads();

    if (thread_group_idx == 0) {
#pragma unroll
        for (int ii = 0; ii < topk; ++ii) {
            const int64_t idx = topk * thread_row + ii;
            output[idx] = output[idx] / output_sum;
            if (apply_routed_scaling_factor_on_output) {
                output[idx] *= routed_scaling_factor;
            }
        }
    }
}

template <
    typename T,
    int VPT,
    int NUM_EXPERTS,
    int THREADS_PER_ROW,
    int ROWS_PER_WARP,
    int ROWS_PER_CTA,
    int WARPS_PER_CTA_>
__global__ void moe_fused_gate_kernel(
    const T *__restrict__ input,
    const float *__restrict__ bias,
    float *__restrict__ output,
    int32_t *__restrict__ indices,
    int64_t num_rows,
    int64_t topk_group,
    int64_t topk,
    int64_t num_fused_shared_experts,
    float routed_scaling_factor,
    bool apply_routed_scaling_factor_on_output) {
    KernelParams<VPT, NUM_EXPERTS, THREADS_PER_ROW, ROWS_PER_WARP, ROWS_PER_CTA, WARPS_PER_CTA_> params;
    moe_fused_gate_impl<T>(
        input, bias, output, indices, num_rows, topk_group, topk, num_fused_shared_experts,
        routed_scaling_factor, apply_routed_scaling_factor_on_output, params);
}

template <typename T>
__global__ void moe_fused_gate_kernel_dynamic(
    const T *__restrict__ input,
    const float *__restrict__ bias,
    float *__restrict__ output,
    int32_t *__restrict__ indices,
    int64_t num_rows,
    int64_t num_experts,
    int64_t num_expert_group,
    int64_t topk_group,
    int64_t topk,
    int64_t num_fused_shared_experts,
    float routed_scaling_factor,
    bool apply_routed_scaling_factor_on_output) {
    KernelParamsDynamic params;
    params.NUM_EXPERTS = static_cast<int>(num_experts);
    params.VPT = static_cast<int>(num_experts / num_expert_group);
    params.THREADS_PER_ROW = static_cast<int>(num_expert_group);
    params.WARPS_PER_CTA = WARPS_PER_CTA;
    params.ROWS_PER_WARP = (num_expert_group > WARP_SIZE) ? 1 : static_cast<int>(WARP_SIZE / num_expert_group);
    params.ROWS_PER_CTA = params.WARPS_PER_CTA * params.ROWS_PER_WARP;
    moe_fused_gate_impl<T>(
        input, bias, output, indices, num_rows, topk_group, topk, num_fused_shared_experts,
        routed_scaling_factor, apply_routed_scaling_factor_on_output, params);
}

#define LAUNCH_MOE_GATE_CONFIG(TYPE, EXPERTS, EXPERT_GROUP)                                                     \
    do {                                                                                                        \
        constexpr int VPT = (EXPERTS) / (EXPERT_GROUP);                                                         \
        constexpr int ROWS_PER_WARP = ((EXPERT_GROUP) <= WARP_SIZE) ? (WARP_SIZE / (EXPERT_GROUP)) : 1;         \
        constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;                                             \
        moe_fused_gate_kernel<TYPE, VPT, (EXPERTS), (EXPERT_GROUP), ROWS_PER_WARP, ROWS_PER_CTA, WARPS_PER_CTA> \
            <<<num_blocks, block_dim, 0, stream>>>(                                                             \
                input_t, bias_t, weights_t, indices_t, num_rows, topk_group, topk,                              \
                num_fused_shared_experts, routed_scaling_factor, apply_routed_scaling_factor_on_output);        \
        dispatched = true;                                                                                      \
    } while (0)

template <typename T>
infiniStatus_t launch(
    const MoeFusedGateInfo &info,
    void *topk_weights,
    void *topk_indices,
    const void *input,
    const void *bias,
    cudaStream_t stream) {
    if (info.num_tokens == 0) {
        return INFINI_STATUS_SUCCESS;
    }
    if ((info.num_experts & (info.num_experts - 1)) != 0 || info.num_experts % info.num_expert_group != 0 || info.num_experts / info.num_expert_group > MAX_VPT) {
        return INFINI_STATUS_BAD_PARAM;
    }

    const auto *input_t = static_cast<const T *>(input);
    const auto *bias_t = static_cast<const float *>(bias);
    auto *weights_t = static_cast<float *>(topk_weights);
    auto *indices_t = static_cast<int32_t *>(topk_indices);

    const int64_t num_rows = static_cast<int64_t>(info.num_tokens);
    const int64_t num_experts = static_cast<int64_t>(info.num_experts);
    const int64_t num_expert_group = static_cast<int64_t>(info.num_expert_group);
    const int64_t topk_group = static_cast<int64_t>(info.topk_group);
    const int64_t topk = static_cast<int64_t>(info.topk);
    const int64_t num_fused_shared_experts = static_cast<int64_t>(info.num_fused_shared_experts);
    const float routed_scaling_factor = info.routed_scaling_factor;
    const bool apply_routed_scaling_factor_on_output = info.apply_routed_scaling_factor_on_output;

    const int64_t rows_per_warp = (num_expert_group > WARP_SIZE) ? 1 : (WARP_SIZE / num_expert_group);
    const int64_t num_warps = (num_rows + rows_per_warp - 1) / rows_per_warp;
    const int64_t num_blocks = (num_warps + WARPS_PER_CTA - 1) / WARPS_PER_CTA;
    dim3 block_dim(WARP_SIZE, WARPS_PER_CTA);

    bool dispatched = false;
    switch (info.num_experts) {
    case 256:
        if (info.num_expert_group == 8) {
            LAUNCH_MOE_GATE_CONFIG(T, 256, 8);
        } else if (info.num_expert_group == 16) {
            LAUNCH_MOE_GATE_CONFIG(T, 256, 16);
        }
        break;
    case 128:
        if (info.num_expert_group == 4) {
            LAUNCH_MOE_GATE_CONFIG(T, 128, 4);
        } else if (info.num_expert_group == 8) {
            LAUNCH_MOE_GATE_CONFIG(T, 128, 8);
        }
        break;
    default:
        break;
    }

    if (!dispatched) {
        moe_fused_gate_kernel_dynamic<T><<<num_blocks, block_dim, 0, stream>>>(
            input_t,
            bias_t,
            weights_t,
            indices_t,
            num_rows,
            num_experts,
            num_expert_group,
            topk_group,
            topk,
            num_fused_shared_experts,
            routed_scaling_factor,
            apply_routed_scaling_factor_on_output);
    }
    return INFINI_STATUS_SUCCESS;
}

#undef LAUNCH_MOE_GATE_CONFIG

} // namespace

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t topk_weights_desc,
    infiniopTensorDescriptor_t topk_indices_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t bias_desc,
    size_t num_expert_group,
    size_t topk_group,
    size_t num_fused_shared_experts,
    float routed_scaling_factor,
    bool apply_routed_scaling_factor_on_output) {
    auto result = MoeFusedGateInfo::create(
        topk_weights_desc,
        topk_indices_desc,
        input_desc,
        bias_desc,
        num_expert_group,
        topk_group,
        num_fused_shared_experts,
        routed_scaling_factor,
        apply_routed_scaling_factor_on_output);
    CHECK_RESULT(result);
    auto info = result.take();
    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        std::move(info),
        0,
        handle->device,
        handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *topk_weights,
    void *topk_indices,
    const void *input,
    const void *bias,
    void *stream) const {
    (void)workspace;
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    switch (_info.dtype) {
    case INFINI_DTYPE_F16:
        return launch<half>(_info, topk_weights, topk_indices, input, bias, cuda_stream);
    case INFINI_DTYPE_BF16:
        return launch<__nv_bfloat16>(_info, topk_weights, topk_indices, input, bias, cuda_stream);
    case INFINI_DTYPE_F32:
        return launch<float>(_info, topk_weights, topk_indices, input, bias, cuda_stream);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::moe_fused_gate::nvidia
