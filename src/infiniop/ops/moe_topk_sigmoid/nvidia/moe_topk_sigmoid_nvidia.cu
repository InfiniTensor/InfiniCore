/*
 * Portions of the CUDA kernels in this file are adapted from SGLang:
 * /sgl-kernel/csrc/moe/moe_topk_sigmoid_kernels.cu
 *
 * Copyright 2025 SGLang Team. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0.
 */

#include "moe_topk_sigmoid_nvidia.cuh"

#include "../../../devices/nvidia/nvidia_common.cuh"

#include <cfloat>
#include <cub/cub.cuh>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <memory>
#include <type_traits>
#include <utility>

namespace op::moe_topk_sigmoid::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

namespace {

constexpr int WARP_SIZE = 32;

template <typename T, int N, int Alignment = sizeof(T) * N>
class alignas(Alignment) AlignedArray {
    T data[N];
};

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

template <typename T, int TPB>
__launch_bounds__(TPB) __global__ void moeSigmoid(
    const T *input,
    const bool *finished,
    float *output,
    const int num_cols,
    const float *correction_bias) {
    const int thread_row_offset = blockIdx.x * num_cols;
    if ((finished != nullptr) && finished[blockIdx.x]) {
        return;
    }
    for (int ii = threadIdx.x; ii < num_cols; ii += TPB) {
        const int idx = thread_row_offset + ii;
        float val = convert_to_float<T>(input[idx]);
        val = 1.0f / (1.0f + expf(-val));
        if (correction_bias != nullptr) {
            val += correction_bias[ii];
        }
        output[idx] = val;
    }
}

template <int TPB>
__launch_bounds__(TPB) __global__ void moeTopK(
    const float *inputs_after_sigmoid,
    const bool *finished,
    float *output,
    int *indices,
    const int num_experts,
    const int k,
    const int start_expert,
    const int end_expert,
    const bool renormalize,
    const float *correction_bias) {
    using cub_kvp = cub::KeyValuePair<int, float>;
    using BlockReduce = cub::BlockReduce<cub_kvp, TPB>;
    __shared__ typename BlockReduce::TempStorage tmp_storage;
    cub_kvp thread_kvp;
    cub::ArgMax arg_max;

    const int block_row = blockIdx.x;
    const bool row_is_active = finished ? !finished[block_row] : true;
    const int thread_read_offset = blockIdx.x * num_experts;
    float row_sum_for_renormalize = 0.0f;

    for (int k_idx = 0; k_idx < k; ++k_idx) {
        thread_kvp.key = 0;
        thread_kvp.value = -1.0f;

        cub_kvp inp_kvp;
        for (int expert = threadIdx.x; expert < num_experts; expert += TPB) {
            const int idx = thread_read_offset + expert;
            inp_kvp.key = expert;
            inp_kvp.value = inputs_after_sigmoid[idx];
            for (int prior_k = 0; prior_k < k_idx; ++prior_k) {
                const int prior_winning_expert = indices[k * block_row + prior_k];
                if (prior_winning_expert == expert) {
                    inp_kvp = thread_kvp;
                }
            }
            thread_kvp = arg_max(inp_kvp, thread_kvp);
        }

        const cub_kvp result_kvp = BlockReduce(tmp_storage).Reduce(thread_kvp, arg_max);
        if (threadIdx.x == 0) {
            const int expert = result_kvp.key;
            const bool node_uses_expert = expert >= start_expert && expert < end_expert;
            const bool should_process_row = row_is_active && node_uses_expert;
            const int idx = k * block_row + k_idx;
            float val = result_kvp.value;
            if (correction_bias != nullptr) {
                val -= correction_bias[expert];
            }
            output[idx] = val;
            indices[idx] = should_process_row ? (expert - start_expert) : num_experts;
            row_sum_for_renormalize += val;
        }
        __syncthreads();
    }

    if (renormalize && threadIdx.x == 0) {
        const float inv = 1.0f / row_sum_for_renormalize;
        for (int k_idx = 0; k_idx < k; ++k_idx) {
            const int idx = k * block_row + k_idx;
            output[idx] *= inv;
        }
    }
}

template <typename T, int VPT, int NUM_EXPERTS, int WARPS_PER_CTA, int BYTES_PER_LDG>
__launch_bounds__(WARPS_PER_CTA *WARP_SIZE) __global__ void topkGatingSigmoid(
    const T *input,
    const bool *finished,
    float *output,
    const int num_rows,
    int *indices,
    const int k,
    const int start_expert,
    const int end_expert,
    const bool renormalize,
    const float *correction_bias) {
    static_assert(VPT == (VPT & -VPT), "VPT must be power of 2");
    static_assert(NUM_EXPERTS == (NUM_EXPERTS & -NUM_EXPERTS), "NUM_EXPERTS must be power of 2");
    static_assert(BYTES_PER_LDG == (BYTES_PER_LDG & -BYTES_PER_LDG), "BYTES_PER_LDG must be power of 2");
    static_assert(BYTES_PER_LDG <= 16, "BYTES_PER_LDG must be leq 16");

    static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(T);
    static constexpr int ELTS_PER_ROW = NUM_EXPERTS;
    static constexpr int THREADS_PER_ROW = ELTS_PER_ROW / VPT;
    static constexpr int LDG_PER_THREAD = VPT / ELTS_PER_LDG;
    static constexpr int ELTS_PER_WARP = WARP_SIZE * VPT;
    static constexpr int ROWS_PER_WARP = ELTS_PER_WARP / ELTS_PER_ROW;
    static constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;
    static constexpr int COLS_PER_GROUP_LDG = ELTS_PER_LDG * THREADS_PER_ROW;

    static_assert(VPT % ELTS_PER_LDG == 0, "VPT must be a multiple of elements per load");
    static_assert(WARP_SIZE % THREADS_PER_ROW == 0, "threads per row must divide warp size");
    static_assert(THREADS_PER_ROW == (THREADS_PER_ROW & -THREADS_PER_ROW), "THREADS_PER_ROW must be power of 2");
    static_assert(THREADS_PER_ROW <= WARP_SIZE, "THREADS_PER_ROW can be at most warp size");
    static_assert(ELTS_PER_WARP % ELTS_PER_ROW == 0, "row elements must divide warp elements");

    const int cta_base_row = blockIdx.x * ROWS_PER_CTA;
    const int warp_base_row = cta_base_row + threadIdx.y * ROWS_PER_WARP;
    const int thread_row_in_warp = threadIdx.x / THREADS_PER_ROW;
    const int thread_row = warp_base_row + thread_row_in_warp;
    if (thread_row >= num_rows) {
        return;
    }
    const bool row_is_active = finished ? !finished[thread_row] : true;

    const T *thread_row_ptr = input + thread_row * ELTS_PER_ROW;
    const int thread_group_idx = threadIdx.x % THREADS_PER_ROW;
    const int first_elt_read_by_thread = thread_group_idx * ELTS_PER_LDG;
    const T *thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;

    using AccessType = AlignedArray<T, ELTS_PER_LDG>;
    T row_chunk_temp[VPT];
    auto *row_chunk_vec_ptr = reinterpret_cast<AccessType *>(&row_chunk_temp);
    const auto *vec_thread_read_ptr = reinterpret_cast<const AccessType *>(thread_read_ptr);
#pragma unroll
    for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
        row_chunk_vec_ptr[ii] = vec_thread_read_ptr[ii * THREADS_PER_ROW];
    }

    float row_chunk[VPT];
#pragma unroll
    for (int ii = 0; ii < VPT; ++ii) {
        float val = convert_to_float<T>(row_chunk_temp[ii]);
        val = 1.0f / (1.0f + expf(-val));
        if (correction_bias != nullptr) {
            const int group_id = ii / ELTS_PER_LDG;
            const int local_id = ii % ELTS_PER_LDG;
            const int expert_idx = first_elt_read_by_thread + group_id * THREADS_PER_ROW * ELTS_PER_LDG + local_id;
            val += correction_bias[expert_idx];
        }
        row_chunk[ii] = val;
    }

    const int start_col = first_elt_read_by_thread;
    float row_sum_for_renormalize = 0.0f;
    for (int k_idx = 0; k_idx < k; ++k_idx) {
        float max_val = row_chunk[0];
        int expert = start_col;
#pragma unroll
        for (int ldg = 0, col = start_col; ldg < LDG_PER_THREAD; ++ldg, col += COLS_PER_GROUP_LDG) {
#pragma unroll
            for (int ii = 0; ii < ELTS_PER_LDG; ++ii) {
                float val = row_chunk[ldg * ELTS_PER_LDG + ii];
                if (val > max_val) {
                    max_val = val;
                    expert = col + ii;
                }
            }
        }

#pragma unroll
        for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
            float other_max = __shfl_xor_sync(0xffffffff, max_val, mask, THREADS_PER_ROW);
            int other_expert = __shfl_xor_sync(0xffffffff, expert, mask, THREADS_PER_ROW);
            if (other_max > max_val || (other_max == max_val && other_expert < expert)) {
                max_val = other_max;
                expert = other_expert;
            }
        }

        if (thread_group_idx == 0) {
            const bool node_uses_expert = expert >= start_expert && expert < end_expert;
            const bool should_process_row = row_is_active && node_uses_expert;
            const int idx = k * thread_row + k_idx;
            if (correction_bias != nullptr) {
                max_val -= correction_bias[expert];
            }
            output[idx] = max_val;
            indices[idx] = should_process_row ? (expert - start_expert) : NUM_EXPERTS;
            row_sum_for_renormalize += max_val;
        }

        if (k_idx + 1 < k) {
            const int ldg_group_for_expert = expert / COLS_PER_GROUP_LDG;
            const int thread_to_clear_in_group = (expert / ELTS_PER_LDG) % THREADS_PER_ROW;
            if (thread_group_idx == thread_to_clear_in_group) {
                const int offset_for_expert = expert % ELTS_PER_LDG;
                row_chunk[ldg_group_for_expert * ELTS_PER_LDG + offset_for_expert] = -10000.0f;
            }
        }
    }

    if (renormalize && thread_group_idx == 0) {
        const float inv = 1.0f / row_sum_for_renormalize;
#pragma unroll
        for (int k_idx = 0; k_idx < k; ++k_idx) {
            const int idx = k * thread_row + k_idx;
            output[idx] *= inv;
        }
    }
}

namespace detail {

template <typename T, int EXPERTS, int BYTES_PER_LDG>
struct TopkConstants {
    static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(T);
    static_assert(EXPERTS / (ELTS_PER_LDG * WARP_SIZE) == 0 || EXPERTS % (ELTS_PER_LDG * WARP_SIZE) == 0, "");
    static constexpr int VECS_PER_THREAD = (EXPERTS / (ELTS_PER_LDG * WARP_SIZE)) > 1 ? (EXPERTS / (ELTS_PER_LDG * WARP_SIZE)) : 1;
    static constexpr int VPT = VECS_PER_THREAD * ELTS_PER_LDG;
    static constexpr int THREADS_PER_ROW = EXPERTS / VPT;
    static constexpr int ROWS_PER_WARP = WARP_SIZE / THREADS_PER_ROW;
};

} // namespace detail

template <typename T, int EXPERTS, int WARPS_PER_TB>
void topkGatingSigmoidLauncherHelper(
    const T *input,
    const bool *finished,
    float *output,
    int *indices,
    const int num_rows,
    const int k,
    const int start_expert,
    const int end_expert,
    const bool renormalize,
    const float *correction_bias,
    cudaStream_t stream) {
    static constexpr int MAX_BYTES_PER_LDG = 16;
    static constexpr int BYTES_PER_LDG = (MAX_BYTES_PER_LDG < static_cast<int>(sizeof(T)) * EXPERTS)
                                           ? MAX_BYTES_PER_LDG
                                           : static_cast<int>(sizeof(T)) * EXPERTS;
    using Constants = detail::TopkConstants<T, EXPERTS, BYTES_PER_LDG>;
    static constexpr int VPT = Constants::VPT;
    static constexpr int ROWS_PER_WARP = Constants::ROWS_PER_WARP;
    const int num_warps = (num_rows + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
    const int num_blocks = (num_warps + WARPS_PER_TB - 1) / WARPS_PER_TB;
    dim3 block_dim(WARP_SIZE, WARPS_PER_TB);
    topkGatingSigmoid<T, VPT, EXPERTS, WARPS_PER_TB, BYTES_PER_LDG><<<num_blocks, block_dim, 0, stream>>>(
        input, finished, output, num_rows, indices, k, start_expert, end_expert,
        renormalize, correction_bias);
}

#define LAUNCH_SIGMOID(TYPE, NUM_EXPERTS, WARPS_PER_TB)               \
    topkGatingSigmoidLauncherHelper<TYPE, NUM_EXPERTS, WARPS_PER_TB>( \
        gating_output, nullptr, topk_weights, topk_indices, num_tokens, topk, 0, num_experts, renormalize, correction_bias, stream)

template <typename T>
infiniStatus_t topkGatingSigmoidKernelLauncher(
    const T *gating_output,
    float *topk_weights,
    int *topk_indices,
    float *sigmoid_workspace,
    const int num_tokens,
    const int num_experts,
    const int topk,
    const bool renormalize,
    const float *correction_bias,
    cudaStream_t stream) {
    static constexpr int WARPS_PER_TB = 4;
    switch (num_experts) {
    case 1:
        LAUNCH_SIGMOID(T, 1, WARPS_PER_TB);
        break;
    case 2:
        LAUNCH_SIGMOID(T, 2, WARPS_PER_TB);
        break;
    case 4:
        LAUNCH_SIGMOID(T, 4, WARPS_PER_TB);
        break;
    case 8:
        LAUNCH_SIGMOID(T, 8, WARPS_PER_TB);
        break;
    case 16:
        LAUNCH_SIGMOID(T, 16, WARPS_PER_TB);
        break;
    case 32:
        LAUNCH_SIGMOID(T, 32, WARPS_PER_TB);
        break;
    case 64:
        LAUNCH_SIGMOID(T, 64, WARPS_PER_TB);
        break;
    case 128:
        LAUNCH_SIGMOID(T, 128, WARPS_PER_TB);
        break;
    case 256:
        LAUNCH_SIGMOID(T, 256, WARPS_PER_TB);
        break;
    default: {
        if (sigmoid_workspace == nullptr) {
            return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
        }
        static constexpr int TPB = 256;
        moeSigmoid<T, TPB><<<num_tokens, TPB, 0, stream>>>(
            gating_output, nullptr, sigmoid_workspace, num_experts, correction_bias);
        moeTopK<TPB><<<num_tokens, TPB, 0, stream>>>(
            sigmoid_workspace, nullptr, topk_weights, topk_indices, num_experts, topk, 0, num_experts,
            renormalize, correction_bias);
        break;
    }
    }
    return INFINI_STATUS_SUCCESS;
}

#undef LAUNCH_SIGMOID

bool needs_workspace(size_t num_experts) {
    const bool is_pow_2 = (num_experts != 0) && ((num_experts & (num_experts - 1)) == 0);
    return !is_pow_2 || num_experts > 256;
}

template <typename T>
infiniStatus_t launch(
    const MoeTopkSigmoidInfo &info,
    void *workspace,
    void *topk_weights,
    void *topk_indices,
    const void *gating_output,
    const void *correction_bias,
    cudaStream_t stream) {
    if (info.num_tokens == 0) {
        return INFINI_STATUS_SUCCESS;
    }
    return topkGatingSigmoidKernelLauncher<T>(
        static_cast<const T *>(gating_output),
        static_cast<float *>(topk_weights),
        static_cast<int *>(topk_indices),
        static_cast<float *>(workspace),
        static_cast<int>(info.num_tokens),
        static_cast<int>(info.num_experts),
        static_cast<int>(info.topk),
        info.renormalize,
        static_cast<const float *>(correction_bias),
        stream);
}

} // namespace

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t topk_weights_desc,
    infiniopTensorDescriptor_t topk_indices_desc,
    infiniopTensorDescriptor_t gating_output_desc,
    infiniopTensorDescriptor_t correction_bias_desc,
    bool renormalize) {
    auto result = MoeTopkSigmoidInfo::create(
        topk_weights_desc,
        topk_indices_desc,
        gating_output_desc,
        correction_bias_desc,
        renormalize);
    CHECK_RESULT(result);
    auto info = result.take();
    const size_t workspace_size = needs_workspace(info.num_experts)
                                    ? info.num_tokens * info.num_experts * sizeof(float)
                                    : 0;
    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        std::move(info),
        workspace_size,
        handle->device,
        handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *topk_weights,
    void *topk_indices,
    const void *gating_output,
    const void *correction_bias,
    void *stream) const {
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    if (_info.has_correction_bias && correction_bias == nullptr) {
        return INFINI_STATUS_BAD_PARAM;
    }
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    switch (_info.dtype) {
    case INFINI_DTYPE_F16:
        return launch<half>(_info, workspace, topk_weights, topk_indices, gating_output, correction_bias, cuda_stream);
    case INFINI_DTYPE_BF16:
        return launch<__nv_bfloat16>(_info, workspace, topk_weights, topk_indices, gating_output, correction_bias, cuda_stream);
    case INFINI_DTYPE_F32:
        return launch<float>(_info, workspace, topk_weights, topk_indices, gating_output, correction_bias, cuda_stream);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::moe_topk_sigmoid::nvidia
