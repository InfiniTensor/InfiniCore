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
    if (candidate_block < 0 || isnan(candidate_score)) {
        return false;
    }
    if (best_block < 0) {
        return true;
    }
    return candidate_score > best_score || (candidate_score == best_score && candidate_block < best_block);
}

__device__ uint32_t ordered_float_key(float value) {
    // The legacy comparator treats +0.0 and -0.0 as equal, then resolves the
    // tie by block id. Canonicalize zero so radix selection preserves that.
    const float normalized = value == 0.0f ? 0.0f : value;
    const uint32_t bits = __float_as_uint(normalized);
    return (bits & 0x80000000U) != 0 ? ~bits : (bits ^ 0x80000000U);
}

template <typename T>
__global__ void deepseek_v4_indexer_score_kernel(
    const T *__restrict__ q,
    const T *__restrict__ weights,
    const T *__restrict__ compressed,
    const long long *__restrict__ positions,
    float *__restrict__ scores,
    size_t batch_size,
    size_t query_len,
    size_t index_n_heads,
    size_t head_dim,
    size_t num_blocks,
    size_t query_start,
    size_t compress_ratio,
    float score_scale,
    float weight_scale) {
    const size_t row_block = blockIdx.x;
    const size_t row = row_block / num_blocks;
    if (row >= batch_size * query_len) {
        return;
    }
    const size_t block = row_block - row * num_blocks;
    const size_t b = row / query_len;
    const size_t tq = row - b * query_len;
    const size_t tid = threadIdx.x;

    const long long pos = positions[query_start + tq];
    long long threshold = (pos + 1) / static_cast<long long>(compress_ratio);
    if (threshold < 0) {
        threshold = 0;
    }
    size_t valid_blocks = static_cast<size_t>(threshold);
    if (valid_blocks > num_blocks) {
        valid_blocks = num_blocks;
    }
    if (block >= valid_blocks) {
        if (tid == 0) {
            scores[row * num_blocks + block] = CUDART_NAN_F;
        }
        return;
    }

    extern __shared__ float head_scores[];
    const size_t k_base = (b * num_blocks + block) * head_dim;
    for (size_t h = tid; h < index_n_heads; h += blockDim.x) {
        float dot = 0.0f;
        const size_t q_base = (row * index_n_heads + h) * head_dim;
        for (size_t d = 0; d < head_dim; ++d) {
            dot += to_float<T>(q[q_base + d]) * to_float<T>(compressed[k_base + d]);
        }
        const float relu_score = fmaxf(0.0f, dot * score_scale);
        head_scores[h] = relu_score
                       * to_float<T>(weights[row * index_n_heads + h])
                       * weight_scale;
    }
    __syncthreads();

    if (tid == 0) {
        float score_sum = 0.0f;
        for (size_t h = 0; h < index_n_heads; ++h) {
            score_sum += head_scores[h];
        }
        scores[row * num_blocks + block] = score_sum;
    }
}

template <int THREADS>
__global__ void deepseek_v4_indexer_select_kernel(
    const float *__restrict__ scores,
    long long *__restrict__ indices,
    size_t rows,
    size_t num_blocks,
    size_t topk) {
    constexpr int RADIX = 256;
    constexpr int MAX_TOPK = 512;
    const size_t row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const int tid = threadIdx.x;
    const float *row_scores = scores + row * num_blocks;
    __shared__ uint32_t histogram[RADIX];
    __shared__ float selected_scores[MAX_TOPK];
    __shared__ int selected_blocks[MAX_TOPK];
    __shared__ uint32_t prefix;
    __shared__ uint32_t prefix_mask;
    __shared__ uint32_t threshold_key;
    __shared__ uint32_t remaining;
    __shared__ uint32_t candidate_count;
    __shared__ bool use_threshold;

    if (tid == 0) {
        prefix = 0;
        prefix_mask = 0;
        threshold_key = 0;
        remaining = static_cast<uint32_t>(topk);
        candidate_count = 0;
        use_threshold = true;
    }
    __syncthreads();

    // Four byte-wise radix passes find the exact K-th ordered float key.
    for (int pass = 0; pass < 4; ++pass) {
        if (tid < RADIX) {
            histogram[tid] = 0;
        }
        __syncthreads();
        const int shift = 24 - pass * 8;
        for (size_t block = static_cast<size_t>(tid); block < num_blocks;
             block += THREADS) {
            const float score = row_scores[block];
            if (isnan(score)) {
                continue;
            }
            const uint32_t key = ordered_float_key(score);
            if ((key & prefix_mask) == prefix) {
                atomicAdd(&histogram[(key >> shift) & 0xffU], 1U);
            }
        }
        __syncthreads();
        if (tid == 0) {
            uint32_t total = 0;
            for (int bin = RADIX - 1; bin >= 0; --bin) {
                total += histogram[bin];
            }
            if (pass == 0 && total <= topk) {
                use_threshold = false;
            } else {
                uint32_t better_count = 0;
                for (int bin = RADIX - 1; bin >= 0; --bin) {
                    const uint32_t count = histogram[bin];
                    if (better_count + count >= remaining) {
                        prefix |= static_cast<uint32_t>(bin) << shift;
                        prefix_mask |= 0xffU << shift;
                        remaining -= better_count;
                        break;
                    }
                    better_count += count;
                }
                if (pass == 3) {
                    threshold_key = prefix;
                }
            }
        }
        __syncthreads();
        if (!use_threshold) {
            break;
        }
    }

    for (size_t block = static_cast<size_t>(tid); block < num_blocks;
         block += THREADS) {
        const float score = row_scores[block];
        if (isnan(score)) {
            continue;
        }
        const uint32_t key = ordered_float_key(score);
        if (!use_threshold || key > threshold_key) {
            const uint32_t slot = atomicAdd(&candidate_count, 1U);
            if (slot < topk) {
                selected_scores[slot] = score;
                selected_blocks[slot] = static_cast<int>(block);
            }
        }
    }
    __syncthreads();

    // Resolve threshold ties in block order to preserve the old deterministic
    // score-descending, lower-block-first behavior.
    if (tid == 0 && use_threshold) {
        uint32_t slot = candidate_count;
        for (size_t block = 0; block < num_blocks && slot < topk; ++block) {
            const float score = row_scores[block];
            if (!isnan(score) && ordered_float_key(score) == threshold_key) {
                selected_scores[slot] = score;
                selected_blocks[slot] = static_cast<int>(block);
                ++slot;
            }
        }
        candidate_count = slot;
    }
    __syncthreads();

    for (size_t slot = static_cast<size_t>(tid); slot < MAX_TOPK;
         slot += THREADS) {
        if (slot >= candidate_count || slot >= topk) {
            selected_scores[slot] = -CUDART_INF_F;
            selected_blocks[slot] = -1;
        }
    }
    __syncthreads();

    // Sort only the fixed 512 candidates instead of rescanning all blocks 512
    // times. Ties retain lower block ids first.
    for (int width = 2; width <= MAX_TOPK; width <<= 1) {
        for (int stride = width >> 1; stride > 0; stride >>= 1) {
            const int slot = (tid / stride) * (stride * 2) + (tid % stride);
            const int partner = slot + stride;
            if (partner < MAX_TOPK) {
                const bool descending_half = (slot & width) == 0;
                const bool partner_is_better = better_pair(
                    selected_scores[partner], selected_blocks[partner],
                    selected_scores[slot], selected_blocks[slot]);
                const bool current_is_better = better_pair(
                    selected_scores[slot], selected_blocks[slot],
                    selected_scores[partner], selected_blocks[partner]);
                if ((descending_half && partner_is_better)
                    || (!descending_half && current_is_better)) {
                    const float score = selected_scores[slot];
                    const int block = selected_blocks[slot];
                    selected_scores[slot] = selected_scores[partner];
                    selected_blocks[slot] = selected_blocks[partner];
                    selected_scores[partner] = score;
                    selected_blocks[partner] = block;
                }
            }
            __syncthreads();
        }
    }

    for (size_t slot = static_cast<size_t>(tid); slot < topk;
         slot += THREADS) {
        indices[row * topk + slot]
            = static_cast<long long>(selected_blocks[slot]);
    }
}

template <typename T>
infiniStatus_t launch_typed(
    const DeepseekV4IndexerInfo &info,
    void *workspace,
    size_t workspace_size,
    long long *indices,
    const void *q,
    const void *weights,
    const void *compressed,
    const void *positions,
    cudaStream_t stream) {
    constexpr int SCORE_THREADS = 256;
    constexpr int SELECT_THREADS = 256;
    constexpr size_t MAX_TOPK = 512;
    const size_t rows = info.batch_size * info.query_len;
    const size_t score_count = rows * info.num_blocks;
    const size_t score_bytes = score_count * sizeof(float);
    if (workspace_size < score_bytes) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    auto *scores = reinterpret_cast<float *>(workspace);
    const size_t score_blocks = score_count;
    const size_t score_shared_bytes = info.index_n_heads * sizeof(float);
    deepseek_v4_indexer_score_kernel<T><<<score_blocks, SCORE_THREADS, score_shared_bytes, stream>>>(
        reinterpret_cast<const T *>(q),
        reinterpret_cast<const T *>(weights),
        reinterpret_cast<const T *>(compressed),
        reinterpret_cast<const long long *>(positions),
        scores,
        info.batch_size,
        info.query_len,
        info.index_n_heads,
        info.head_dim,
        info.num_blocks,
        info.query_start,
        info.compress_ratio,
        info.score_scale,
        info.weight_scale);
    if (info.topk > MAX_TOPK) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }
    deepseek_v4_indexer_select_kernel<SELECT_THREADS>
        <<<rows, SELECT_THREADS, 0, stream>>>(
        scores,
        indices,
        rows,
        info.num_blocks,
        info.topk);
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
    const size_t workspace_size = info.batch_size * info.query_len
                                * info.num_blocks * sizeof(float);
    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        info,
        workspace_size,
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
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    auto stream = reinterpret_cast<cudaStream_t>(stream_);
    if (_info.dtype == INFINI_DTYPE_F16) {
        return launch_typed<half>(_info, workspace, workspace_size,
                                  reinterpret_cast<long long *>(indices), q, weights, compressed, positions, stream);
    }
    if (_info.dtype == INFINI_DTYPE_BF16) {
        return launch_typed<__nv_bfloat16>(_info, workspace, workspace_size,
                                           reinterpret_cast<long long *>(indices), q, weights, compressed, positions, stream);
    }
    if (_info.dtype == INFINI_DTYPE_F32) {
        return launch_typed<float>(_info, workspace, workspace_size,
                                   reinterpret_cast<long long *>(indices), q, weights, compressed, positions, stream);
    }
    return INFINI_STATUS_BAD_TENSOR_DTYPE;
}

} // namespace op::deepseek_v4_indexer::nvidia

#endif
