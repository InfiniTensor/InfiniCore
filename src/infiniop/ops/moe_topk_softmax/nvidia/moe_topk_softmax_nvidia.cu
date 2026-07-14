/*
 * Portions of the CUDA kernels in this file are adapted from SGLang:
 * /sgl-kernel/csrc/moe/moe_topk_softmax_kernels.cu
 *
 * Copyright 2025 SGLang Team. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0.
 */

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_HYGON_API)

#include "moe_topk_softmax_nvidia.cuh"

#include "../../../devices/nvidia/nvidia_common.cuh"

#include <cfloat>
#include <cub/cub.cuh>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <memory>
#include <type_traits>
#include <utility>

namespace op::moe_topk_softmax::nvidia {

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

struct MaxReduceOp {
    __device__ __host__ float operator()(float a, float b) const {
        return a > b ? a : b;
    }
};

template <typename T, int TPB>
__launch_bounds__(TPB) __global__ void moeSoftmax(
    const T *input,
    float *output,
    const int num_cols,
    const float moe_softcapping) {
    using BlockReduce = cub::BlockReduce<float, TPB>;
    __shared__ typename BlockReduce::TempStorage tmp_storage;
    __shared__ float normalizing_factor;
    __shared__ float float_max;

    const int thread_row_offset = blockIdx.x * num_cols;

    float thread_data = -FLT_MAX;
    for (int ii = threadIdx.x; ii < num_cols; ii += TPB) {
        const int idx = thread_row_offset + ii;
        float val = convert_to_float<T>(input[idx]);
        if (moe_softcapping != 0.0f) {
            val = tanhf(val / moe_softcapping) * moe_softcapping;
        }
        output[idx] = val;
        thread_data = fmaxf(val, thread_data);
    }

    const float max_elem = BlockReduce(tmp_storage).Reduce(thread_data, MaxReduceOp());
    if (threadIdx.x == 0) {
        float_max = max_elem;
    }
    __syncthreads();

    thread_data = 0.0f;
    for (int ii = threadIdx.x; ii < num_cols; ii += TPB) {
        const int idx = thread_row_offset + ii;
        thread_data += expf(output[idx] - float_max);
    }
    const float z = BlockReduce(tmp_storage).Sum(thread_data);
    if (threadIdx.x == 0) {
        normalizing_factor = 1.0f / z;
    }
    __syncthreads();

    for (int ii = threadIdx.x; ii < num_cols; ii += TPB) {
        const int idx = thread_row_offset + ii;
        output[idx] = expf(output[idx] - float_max) * normalizing_factor;
    }
}

namespace moe {

using cub_kvp = cub::KeyValuePair<int, float>;

struct TopKPair {
    static const int PAIR = 2;
    static const int MAX_INDEX = 0;
    cub_kvp max;
    cub_kvp secondMax;

    __device__ TopKPair() {}
    __device__ TopKPair(cub_kvp max, cub_kvp secondMax) : max(max), secondMax(secondMax) {}
};

struct TopKPairArgMax {
    __device__ TopKPairArgMax() {}
    __device__ __forceinline__ TopKPair operator()(const TopKPair &candidate1, const TopKPair &candidate2) const {
        cub_kvp globalMax, globalSecondMax;
        if (candidate1.max.value > candidate2.max.value) {
            globalMax = candidate1.max;
        } else {
            globalMax = candidate2.max;
        }
        if (globalMax.key == candidate1.max.key) {
            globalSecondMax = (candidate1.secondMax.value > candidate2.max.value) ? candidate1.secondMax : candidate2.max;
        } else {
            globalSecondMax = (candidate2.secondMax.value > candidate1.max.value) ? candidate2.secondMax : candidate1.max;
        }
        return TopKPair(globalMax, globalSecondMax);
    }
};

} // namespace moe

template <int TPB>
__launch_bounds__(TPB) __global__ void moeTopKFast(
    float *inputs_after_softmax,
    float *output,
    int *indices,
    const int num_experts,
    const int k,
    const bool renormalize,
    const float *correction_bias) {
    using namespace moe;
    using BlockReduce = cub::BlockReduce<TopKPair, TPB>;
    __shared__ typename BlockReduce::TempStorage tmp_storage;
    TopKPair thread_pair;

    const int block_row = blockIdx.x;
    const int thread_read_offset = blockIdx.x * num_experts;
    float row_sum_for_renormalize = 0.0f;

    for (int k_idx = 0; k_idx < (k + TopKPair::PAIR - 1) / TopKPair::PAIR; ++k_idx) {
        thread_pair.max.key = 0;
        thread_pair.max.value = -1.0f;
        thread_pair.secondMax.key = 0;
        thread_pair.secondMax.value = -1.0f;

        cub_kvp inp_kvp;
        for (int expert = threadIdx.x; expert < num_experts; expert += TPB) {
            const int idx = thread_read_offset + expert;
            inp_kvp.key = expert;
            const float prob = inputs_after_softmax[idx];
            inp_kvp.value = correction_bias == nullptr ? prob : prob + correction_bias[expert];
            if (inp_kvp.value > thread_pair.max.value) {
                thread_pair.secondMax = thread_pair.max;
                thread_pair.max = inp_kvp;
            } else if (inp_kvp.value > thread_pair.secondMax.value) {
                thread_pair.secondMax = inp_kvp;
            }
        }

        TopKPairArgMax reducer;
        const TopKPair result_pair = BlockReduce(tmp_storage).Reduce(thread_pair, reducer);
        if (threadIdx.x == 0) {
#if !defined(ENABLE_ILUVATAR_API) && !defined(ENABLE_HYGON_API)
#pragma unroll
#endif
            for (int i = 0; i < TopKPair::PAIR; ++i) {
                if (k_idx * 2 + i >= k) {
                    break;
                }
                cub_kvp result = (i == TopKPair::MAX_INDEX) ? result_pair.max : result_pair.secondMax;
                int expert = result.key;
                const float prob = inputs_after_softmax[thread_read_offset + expert];
                inputs_after_softmax[thread_read_offset + expert] = -FLT_MAX;
                int idx = k * block_row + k_idx * 2 + i;
                output[idx] = prob;
                indices[idx] = expert;
                row_sum_for_renormalize += prob;
            }
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

template <int TPB>
__launch_bounds__(TPB) __global__ void moeTopK(
    float *inputs_after_softmax,
    float *output,
    int *indices,
    const int num_experts,
    const int k,
    const bool renormalize,
    const float *correction_bias) {
    using cub_kvp = cub::KeyValuePair<int, float>;
    using BlockReduce = cub::BlockReduce<cub_kvp, TPB>;
    __shared__ typename BlockReduce::TempStorage tmp_storage;
    cub_kvp thread_kvp;
    cub::ArgMax arg_max;

    const int block_row = blockIdx.x;
    const int thread_read_offset = blockIdx.x * num_experts;
    float row_sum_for_renormalize = 0.0f;

    for (int k_idx = 0; k_idx < k; ++k_idx) {
        thread_kvp.key = 0;
        thread_kvp.value = -1.0f;
        cub_kvp inp_kvp;
        for (int expert = threadIdx.x; expert < num_experts; expert += TPB) {
            const int idx = thread_read_offset + expert;
            inp_kvp.key = expert;
            const float prob = inputs_after_softmax[idx];
            inp_kvp.value = correction_bias == nullptr ? prob : prob + correction_bias[expert];
            thread_kvp = arg_max(inp_kvp, thread_kvp);
        }

        const cub_kvp result_kvp = BlockReduce(tmp_storage).Reduce(thread_kvp, arg_max);
        if (threadIdx.x == 0) {
            const int expert = result_kvp.key;
            const int idx = k * block_row + k_idx;
            const float prob = inputs_after_softmax[thread_read_offset + expert];
            output[idx] = prob;
            indices[idx] = expert;
            row_sum_for_renormalize += prob;
            inputs_after_softmax[thread_read_offset + expert] = -FLT_MAX;
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
__launch_bounds__(WARPS_PER_CTA *WARP_SIZE) __global__ void topkGatingSoftmax(
    const T *input,
    float *output,
    const int num_rows,
    int *indices,
    const int k,
    const bool renormalize,
    const float moe_softcapping,
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

    const T *thread_row_ptr = input + thread_row * ELTS_PER_ROW;
    const int thread_group_idx = threadIdx.x % THREADS_PER_ROW;
    const int first_elt_read_by_thread = thread_group_idx * ELTS_PER_LDG;
    const T *thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;

    using AccessType = AlignedArray<T, ELTS_PER_LDG>;
    T row_chunk_temp[VPT];
    auto *row_chunk_vec_ptr = reinterpret_cast<AccessType *>(&row_chunk_temp);
    const auto *vec_thread_read_ptr = reinterpret_cast<const AccessType *>(thread_read_ptr);
#if !defined(ENABLE_ILUVATAR_API) && !defined(ENABLE_HYGON_API)
#pragma unroll
#endif
    for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
        row_chunk_vec_ptr[ii] = vec_thread_read_ptr[ii * THREADS_PER_ROW];
    }

    float row_chunk[VPT];
#if !defined(ENABLE_ILUVATAR_API) && !defined(ENABLE_HYGON_API)
#pragma unroll
#endif
    for (int ii = 0; ii < VPT; ++ii) {
        row_chunk[ii] = convert_to_float<T>(row_chunk_temp[ii]);
    }

    if (moe_softcapping != 0.0f) {
#if !defined(ENABLE_ILUVATAR_API) && !defined(ENABLE_HYGON_API)
#pragma unroll
#endif
        for (int ii = 0; ii < VPT; ++ii) {
            float val = row_chunk[ii];
            if (moe_softcapping != 0.0f) {
                val = tanhf(val / moe_softcapping) * moe_softcapping;
            }
            row_chunk[ii] = val;
        }
    }

    float thread_max = row_chunk[0];
#if !defined(ENABLE_ILUVATAR_API) && !defined(ENABLE_HYGON_API)
#pragma unroll
#endif
    for (int ii = 1; ii < VPT; ++ii) {
        thread_max = fmaxf(thread_max, row_chunk[ii]);
    }
#if !defined(ENABLE_ILUVATAR_API) && !defined(ENABLE_HYGON_API)
#pragma unroll
#endif
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
        thread_max = fmaxf(thread_max, __shfl_xor_sync(0xffffffff, thread_max, mask, THREADS_PER_ROW));
    }

    float row_sum = 0.0f;
#if !defined(ENABLE_ILUVATAR_API) && !defined(ENABLE_HYGON_API)
#pragma unroll
#endif
    for (int ii = 0; ii < VPT; ++ii) {
        row_chunk[ii] = expf(row_chunk[ii] - thread_max);
        row_sum += row_chunk[ii];
    }
#if !defined(ENABLE_ILUVATAR_API) && !defined(ENABLE_HYGON_API)
#pragma unroll
#endif
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
        row_sum += __shfl_xor_sync(0xffffffff, row_sum, mask, THREADS_PER_ROW);
    }
    const float reciprocal_row_sum = 1.0f / row_sum;
#if !defined(ENABLE_ILUVATAR_API) && !defined(ENABLE_HYGON_API)
#pragma unroll
#endif
    for (int ii = 0; ii < VPT; ++ii) {
        row_chunk[ii] *= reciprocal_row_sum;
    }

    const int start_col = first_elt_read_by_thread;
    float row_sum_for_renormalize = 0.0f;
    for (int k_idx = 0; k_idx < k; ++k_idx) {
        float max_prob = row_chunk[0];
        float max_choice = correction_bias == nullptr ? max_prob : max_prob + correction_bias[start_col];
        int expert = start_col;
#if !defined(ENABLE_ILUVATAR_API) && !defined(ENABLE_HYGON_API)
#pragma unroll
#endif
        for (int ldg = 0, col = start_col; ldg < LDG_PER_THREAD; ++ldg, col += COLS_PER_GROUP_LDG) {
#if !defined(ENABLE_ILUVATAR_API) && !defined(ENABLE_HYGON_API)
#pragma unroll
#endif
            for (int ii = 0; ii < ELTS_PER_LDG; ++ii) {
                const int expert_idx = col + ii;
                float prob = row_chunk[ldg * ELTS_PER_LDG + ii];
                float choice = correction_bias == nullptr ? prob : prob + correction_bias[expert_idx];
                if (choice > max_choice) {
                    max_choice = choice;
                    max_prob = prob;
                    expert = expert_idx;
                }
            }
        }

#if !defined(ENABLE_ILUVATAR_API) && !defined(ENABLE_HYGON_API)
#pragma unroll
#endif
        for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
            float other_choice = __shfl_xor_sync(0xffffffff, max_choice, mask, THREADS_PER_ROW);
            float other_prob = __shfl_xor_sync(0xffffffff, max_prob, mask, THREADS_PER_ROW);
            int other_expert = __shfl_xor_sync(0xffffffff, expert, mask, THREADS_PER_ROW);
            if (other_choice > max_choice || (other_choice == max_choice && other_expert < expert)) {
                max_choice = other_choice;
                max_prob = other_prob;
                expert = other_expert;
            }
        }

        if (thread_group_idx == 0) {
            const int idx = k * thread_row + k_idx;
            output[idx] = max_prob;
            indices[idx] = expert;
            row_sum_for_renormalize += max_prob;
        }

        if (k_idx + 1 < k) {
            const int ldg_group_for_expert = expert / COLS_PER_GROUP_LDG;
            const int thread_to_clear_in_group = (expert / ELTS_PER_LDG) % THREADS_PER_ROW;
            if (thread_group_idx == thread_to_clear_in_group) {
                const int offset_for_expert = expert % ELTS_PER_LDG;
                row_chunk[ldg_group_for_expert * ELTS_PER_LDG + offset_for_expert] = -FLT_MAX;
            }
        }
    }

    if (renormalize && thread_group_idx == 0) {
        const float inv = 1.0f / row_sum_for_renormalize;
#if !defined(ENABLE_ILUVATAR_API) && !defined(ENABLE_HYGON_API)
#pragma unroll
#endif
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
void topkGatingSoftmaxLauncherHelper(
    const T *input,
    float *output,
    int *indices,
    const int num_rows,
    const int k,
    const bool renormalize,
    const float moe_softcapping,
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
    topkGatingSoftmax<T, VPT, EXPERTS, WARPS_PER_TB, BYTES_PER_LDG><<<num_blocks, block_dim, 0, stream>>>(
        input, output, num_rows, indices, k, renormalize, moe_softcapping, correction_bias);
}

#define LAUNCH_SOFTMAX(TYPE, NUM_EXPERTS, WARPS_PER_TB)               \
    topkGatingSoftmaxLauncherHelper<TYPE, NUM_EXPERTS, WARPS_PER_TB>( \
        gating_output, topk_weights, topk_indices, num_tokens, topk, renormalize, moe_softcapping, correction_bias, stream)

template <typename T>
infiniStatus_t topkGatingSoftmaxKernelLauncher(
    const T *gating_output,
    float *topk_weights,
    int *topk_indices,
    float *softmax_workspace,
    const int num_tokens,
    const int num_experts,
    const int topk,
    const bool renormalize,
    const float moe_softcapping,
    const float *correction_bias,
    cudaStream_t stream) {
    static constexpr int WARPS_PER_TB = 4;
    switch (num_experts) {
    case 1:
        LAUNCH_SOFTMAX(T, 1, WARPS_PER_TB);
        break;
    case 2:
        LAUNCH_SOFTMAX(T, 2, WARPS_PER_TB);
        break;
    case 4:
        LAUNCH_SOFTMAX(T, 4, WARPS_PER_TB);
        break;
    case 8:
        LAUNCH_SOFTMAX(T, 8, WARPS_PER_TB);
        break;
    case 16:
        LAUNCH_SOFTMAX(T, 16, WARPS_PER_TB);
        break;
    case 32:
        LAUNCH_SOFTMAX(T, 32, WARPS_PER_TB);
        break;
    case 64:
        LAUNCH_SOFTMAX(T, 64, WARPS_PER_TB);
        break;
    case 128:
        LAUNCH_SOFTMAX(T, 128, WARPS_PER_TB);
        break;
    case 256:
        LAUNCH_SOFTMAX(T, 256, WARPS_PER_TB);
        break;
    case 512:
        LAUNCH_SOFTMAX(T, 512, WARPS_PER_TB);
        break;
    default: {
        if (softmax_workspace == nullptr) {
            return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
        }
        static constexpr int TPB = 256;
        moeSoftmax<T, TPB><<<num_tokens, TPB, 0, stream>>>(
            gating_output, softmax_workspace, num_experts, moe_softcapping);
        if (topk == 1) {
            moeTopK<TPB><<<num_tokens, TPB, 0, stream>>>(
                softmax_workspace, topk_weights, topk_indices, num_experts, topk, renormalize, correction_bias);
        } else {
            moeTopKFast<TPB><<<num_tokens, TPB, 0, stream>>>(
                softmax_workspace, topk_weights, topk_indices, num_experts, topk, renormalize, correction_bias);
        }
        break;
    }
    }
    return INFINI_STATUS_SUCCESS;
}

#undef LAUNCH_SOFTMAX

bool needs_workspace(size_t num_experts) {
    const bool is_pow_2 = (num_experts != 0) && ((num_experts & (num_experts - 1)) == 0);
    return !is_pow_2 || num_experts > 512;
}

template <typename T>
infiniStatus_t launch(
    const MoeTopkSoftmaxInfo &info,
    void *workspace,
    void *topk_weights,
    void *topk_indices,
    const void *gating_output,
    const void *correction_bias,
    cudaStream_t stream) {
    if (info.num_tokens == 0) {
        return INFINI_STATUS_SUCCESS;
    }
    return topkGatingSoftmaxKernelLauncher<T>(
        static_cast<const T *>(gating_output),
        static_cast<float *>(topk_weights),
        static_cast<int *>(topk_indices),
        static_cast<float *>(workspace),
        static_cast<int>(info.num_tokens),
        static_cast<int>(info.num_experts),
        static_cast<int>(info.topk),
        info.renormalize,
        info.moe_softcapping,
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
    bool renormalize,
    float moe_softcapping) {
    auto result = MoeTopkSoftmaxInfo::create(
        topk_weights_desc,
        topk_indices_desc,
        gating_output_desc,
        correction_bias_desc,
        renormalize,
        moe_softcapping);
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

} // namespace op::moe_topk_softmax::nvidia

#endif // ENABLE_NVIDIA_API || ENABLE_HYGON_API
