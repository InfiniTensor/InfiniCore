#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_HYGON_API) || defined(ENABLE_ALI_API)

#include "deepseek_v4_indexer_nvidia.cuh"

#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <memory>
#include <type_traits>

namespace op::deepseek_v4_indexer::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

namespace {

template <typename T>
__device__ float to_float(T value) {
    if constexpr (std::is_same_v<T, half>) {
        return __half2float(value);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return __bfloat162float(value);
    } else {
        return static_cast<float>(value);
    }
}

__device__ bool better_pair(float candidate_score, int candidate_block, float best_score, int best_block) {
    if (candidate_block < 0) {
        return false;
    }
    if (best_block < 0) {
        return true;
    }
    return candidate_score > best_score || (candidate_score == best_score && candidate_block < best_block);
}

template <typename T, int THREADS>
__global__ void deepseek_v4_indexer_kernel(
    const T *__restrict__ q,
    const T *__restrict__ weights,
    const T *__restrict__ compressed,
    const long long *__restrict__ positions,
    long long *__restrict__ indices,
    size_t batch_size,
    size_t query_len,
    size_t index_n_heads,
    size_t head_dim,
    size_t num_blocks,
    size_t topk,
    size_t query_start,
    size_t compress_ratio,
    float score_scale,
    float weight_scale) {
    const size_t row = blockIdx.x;
    if (row >= batch_size * query_len) {
        return;
    }
    const size_t b = row / query_len;
    const size_t tq = row - b * query_len;
    const int tid = threadIdx.x;

    extern __shared__ long long selected_blocks[];
    __shared__ float reduce_scores[THREADS];
    __shared__ int reduce_blocks[THREADS];

    const long long pos = positions[query_start + tq];
    long long threshold = (pos + 1) / static_cast<long long>(compress_ratio);
    if (threshold < 0) {
        threshold = 0;
    }
    size_t valid_blocks = static_cast<size_t>(threshold);
    if (valid_blocks > num_blocks) {
        valid_blocks = num_blocks;
    }

    for (size_t k = 0; k < topk; ++k) {
        float thread_best_score = -CUDART_INF_F;
        int thread_best_block = -1;
        for (size_t block = static_cast<size_t>(tid); block < valid_blocks; block += THREADS) {
            bool already_selected = false;
            for (size_t prev = 0; prev < k; ++prev) {
                if (selected_blocks[prev] == static_cast<long long>(block)) {
                    already_selected = true;
                    break;
                }
            }
            if (already_selected) {
                continue;
            }

            float score_sum = 0.0f;
            for (size_t h = 0; h < index_n_heads; ++h) {
                float dot = 0.0f;
                const size_t q_base = ((b * query_len + tq) * index_n_heads + h) * head_dim;
                const size_t k_base = (b * num_blocks + block) * head_dim;
                for (size_t d = 0; d < head_dim; ++d) {
                    dot += to_float<T>(q[q_base + d]) * to_float<T>(compressed[k_base + d]);
                }
                const float relu_score = fmaxf(0.0f, dot * score_scale);
                score_sum += relu_score * to_float<T>(weights[(b * query_len + tq) * index_n_heads + h]) * weight_scale;
            }
            if (better_pair(score_sum, static_cast<int>(block), thread_best_score, thread_best_block)) {
                thread_best_score = score_sum;
                thread_best_block = static_cast<int>(block);
            }
        }

        reduce_scores[tid] = thread_best_score;
        reduce_blocks[tid] = thread_best_block;
        __syncthreads();
        for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                const float other_score = reduce_scores[tid + stride];
                const int other_block = reduce_blocks[tid + stride];
                if (better_pair(other_score, other_block, reduce_scores[tid], reduce_blocks[tid])) {
                    reduce_scores[tid] = other_score;
                    reduce_blocks[tid] = other_block;
                }
            }
            __syncthreads();
        }

        if (tid == 0) {
            const int winner = reduce_blocks[0];
            selected_blocks[k] = static_cast<long long>(winner);
            indices[row * topk + k] = winner >= 0 ? static_cast<long long>(winner) : -1LL;
        }
        __syncthreads();
    }
}

template <typename T>
infiniStatus_t launch_typed(
    const DeepseekV4IndexerInfo &info,
    long long *indices,
    const void *q,
    const void *weights,
    const void *compressed,
    const void *positions,
    cudaStream_t stream) {
    constexpr int THREADS = 256;
    const size_t rows = info.batch_size * info.query_len;
    const size_t shared_bytes = info.topk * sizeof(long long);
    deepseek_v4_indexer_kernel<T, THREADS><<<rows, THREADS, shared_bytes, stream>>>(
        reinterpret_cast<const T *>(q),
        reinterpret_cast<const T *>(weights),
        reinterpret_cast<const T *>(compressed),
        reinterpret_cast<const long long *>(positions),
        indices,
        info.batch_size,
        info.query_len,
        info.index_n_heads,
        info.head_dim,
        info.num_blocks,
        info.topk,
        info.query_start,
        info.compress_ratio,
        info.score_scale,
        info.weight_scale);
    CHECK_CUDA(cudaGetLastError());
    return INFINI_STATUS_SUCCESS;
}

} // namespace

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t indices_desc,
    infiniopTensorDescriptor_t q_desc,
    infiniopTensorDescriptor_t weights_desc,
    infiniopTensorDescriptor_t compressed_desc,
    infiniopTensorDescriptor_t positions_desc,
    size_t query_start,
    size_t compress_ratio) {
    auto result = DeepseekV4IndexerInfo::create(
        indices_desc, q_desc, weights_desc, compressed_desc, positions_desc, query_start, compress_ratio);
    CHECK_RESULT(result);
    auto info = result.take();
    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        info,
        0,
        handle->device,
        handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *indices,
    const void *q,
    const void *weights,
    const void *compressed,
    const void *positions,
    void *stream_) const {
    (void)workspace;
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    auto stream = reinterpret_cast<cudaStream_t>(stream_);
    if (_info.dtype == INFINI_DTYPE_F16) {
        return launch_typed<half>(_info, reinterpret_cast<long long *>(indices), q, weights, compressed, positions, stream);
    }
    if (_info.dtype == INFINI_DTYPE_BF16) {
        return launch_typed<__nv_bfloat16>(_info, reinterpret_cast<long long *>(indices), q, weights, compressed, positions, stream);
    }
    if (_info.dtype == INFINI_DTYPE_F32) {
        return launch_typed<float>(_info, reinterpret_cast<long long *>(indices), q, weights, compressed, positions, stream);
    }
    return INFINI_STATUS_BAD_TENSOR_DTYPE;
}

} // namespace op::deepseek_v4_indexer::nvidia

#endif
