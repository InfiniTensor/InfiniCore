#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_HYGON_API) || defined(ENABLE_ALI_API)

#include "deepseek_v4_compressed_decode_nvidia.cuh"

#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <type_traits>

namespace op::deepseek_v4_compressed_decode::nvidia {
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

template <typename T>
__device__ T from_float(float value) {
    if constexpr (std::is_same_v<T, half>) {
        return __float2half(value);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return __float2bfloat16(value);
    } else {
        return static_cast<T>(value);
    }
}

template <typename IntT>
__device__ int64_t read_int(const void *values, size_t idx) {
    return static_cast<int64_t>(reinterpret_cast<const IntT *>(values)[idx]);
}

template <typename PosT>
__device__ int64_t read_pos(const void *positions, size_t idx) {
    return read_int<PosT>(positions, idx);
}

template <typename IndexT>
__device__ bool block_selected(const void *indexed_blocks,
                               size_t index_top_k,
                               size_t query_len,
                               size_t batch_idx,
                               size_t query_idx,
                               size_t block) {
    if (index_top_k == 0) {
        return true;
    }
    const size_t base = (batch_idx * query_len + query_idx) * index_top_k;
    const int64_t target = static_cast<int64_t>(block);
    for (size_t i = 0; i < index_top_k; ++i) {
        if (read_int<IndexT>(indexed_blocks, base + i) == target) {
            return true;
        }
    }
    return false;
}

__device__ double yarn_inv_freq(size_t pair_idx,
                                size_t rope_dim,
                                double base,
                                double factor,
                                double beta_fast,
                                double beta_slow,
                                int64_t original_seq_len,
                                double extrapolation_factor) {
    const double inv_freq_extrapolation = 1.0 / pow(base, static_cast<double>(2 * pair_idx) / static_cast<double>(rope_dim));
    if (factor <= 1.0 || original_seq_len <= 0 || base <= 1.0) {
        return inv_freq_extrapolation;
    }
    constexpr double pi = 3.141592653589793238462643383279502884;
    auto correction_dim = [&](double num_rotations) {
        return static_cast<double>(rope_dim)
             * log(static_cast<double>(original_seq_len) / (num_rotations * 2.0 * pi))
             / (2.0 * log(base));
    };
    double low = floor(correction_dim(beta_fast));
    double high = ceil(correction_dim(beta_slow));
    low = fmax(low, 0.0);
    high = fmin(high, static_cast<double>(rope_dim - 1));
    if (low == high) {
        high += 0.001;
    }
    const double ramp = fmin(fmax((static_cast<double>(pair_idx) - low) / (high - low), 0.0), 1.0);
    const double inv_freq_mask = (1.0 - ramp) * extrapolation_factor;
    const double inv_freq_interpolation = inv_freq_extrapolation / factor;
    return inv_freq_interpolation * (1.0 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask;
}

__device__ double rope_inv_freq(size_t pair,
                                size_t rope_dim,
                                double rope_theta,
                                bool use_yarn,
                                double yarn_factor,
                                double yarn_beta_fast,
                                double yarn_beta_slow,
                                int64_t yarn_original_seq_len,
                                double yarn_extrapolation_factor) {
    if (use_yarn) {
        return yarn_inv_freq(pair, rope_dim, rope_theta, yarn_factor, yarn_beta_fast,
                             yarn_beta_slow, yarn_original_seq_len, yarn_extrapolation_factor);
    }
    return 1.0 / pow(rope_theta, static_cast<double>(2 * pair) / static_cast<double>(rope_dim));
}

template <typename T>
__device__ float rotated_comp_value(const T *__restrict__ kv_comp,
                                    size_t base,
                                    size_t d,
                                    size_t head_dim,
                                    size_t rope_dim,
                                    int64_t block_pos,
                                    double rope_theta,
                                    bool use_yarn,
                                    double yarn_factor,
                                    double yarn_beta_fast,
                                    double yarn_beta_slow,
                                    int64_t yarn_original_seq_len,
                                    double yarn_extrapolation_factor) {
    const size_t pass_dim = head_dim - rope_dim;
    if (d < pass_dim) {
        return to_float<T>(kv_comp[base + d]);
    }
    const size_t pair = (d - pass_dim) / 2;
    const size_t even = pass_dim + 2 * pair;
    const size_t odd = even + 1;
    const float raw_even = to_float<T>(kv_comp[base + even]);
    const float raw_odd = to_float<T>(kv_comp[base + odd]);
    const double inv_freq = rope_inv_freq(pair, rope_dim, rope_theta, use_yarn, yarn_factor,
                                          yarn_beta_fast, yarn_beta_slow,
                                          yarn_original_seq_len, yarn_extrapolation_factor);
    const double angle = static_cast<double>(block_pos) * inv_freq;
    const float c = static_cast<float>(cos(angle));
    const float s = static_cast<float>(sin(angle));
    if (((d - pass_dim) & 1) == 0) {
        return raw_even * c - raw_odd * s;
    }
    return raw_odd * c + raw_even * s;
}

template <typename T, typename SinkT, typename PosT, typename IndexT>
__global__ void compressed_decode_kernel(T *__restrict__ y,
                                         const T *__restrict__ q,
                                         const T *__restrict__ k,
                                         const T *__restrict__ kv_comp,
                                         const SinkT *__restrict__ attn_sink,
                                         const void *__restrict__ query_positions,
                                         const void *__restrict__ block_positions,
                                         const void *__restrict__ indexed_blocks,
                                         size_t batch_size,
                                         size_t query_len,
                                         size_t num_heads,
                                         size_t key_len,
                                         size_t num_kv_heads,
                                         size_t num_blocks,
                                         size_t head_dim,
                                         size_t index_top_k,
                                         float softmax_scale,
                                         size_t compress_ratio,
                                         size_t rope_dim,
                                         double rope_theta,
                                         bool use_yarn,
                                         double yarn_factor,
                                         double yarn_beta_fast,
                                         double yarn_beta_slow,
                                         int64_t yarn_original_seq_len,
                                         double yarn_extrapolation_factor) {
    extern __shared__ float smem[];
    const size_t total_keys = num_blocks + key_len;
    float *logits = smem;
    float *scratch = smem + total_keys;

    const size_t row = blockIdx.x;
    const size_t tid = threadIdx.x;
    const size_t rows = batch_size * query_len * num_heads;
    if (row >= rows) {
        return;
    }

    const size_t h = row % num_heads;
    const size_t tmp = row / num_heads;
    const size_t tq = tmp % query_len;
    const size_t b = tmp / query_len;
    const size_t kv_head = h / (num_heads / num_kv_heads);
    const int64_t q_pos = read_pos<PosT>(query_positions, tq);
    const size_t visible_blocks = static_cast<size_t>(max(static_cast<int64_t>(0), (q_pos + 1) / static_cast<int64_t>(compress_ratio)));

    const size_t q_base = ((b * query_len + tq) * num_heads + h) * head_dim;
    const size_t k_base = (b * key_len * num_kv_heads + kv_head) * head_dim;
    const size_t comp_batch_base = b * num_blocks * head_dim;

    float row_max = to_float<SinkT>(attn_sink[h]);
    for (size_t block = 0; block < num_blocks; ++block) {
        float logit = -CUDART_INF_F;
        if (block < visible_blocks && block_selected<IndexT>(indexed_blocks, index_top_k, query_len, b, tq, block)) {
            const size_t comp_base = comp_batch_base + block * head_dim;
            const int64_t block_pos = read_pos<PosT>(block_positions, block);
            float local_dot = 0.0f;
            for (size_t hd = tid; hd < head_dim; hd += blockDim.x) {
                local_dot += to_float<T>(q[q_base + hd]) * rotated_comp_value<T>(
                    kv_comp, comp_base, hd, head_dim, rope_dim, block_pos, rope_theta,
                    use_yarn, yarn_factor, yarn_beta_fast, yarn_beta_slow,
                    yarn_original_seq_len, yarn_extrapolation_factor);
            }
            scratch[tid] = local_dot;
            __syncthreads();
            for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    scratch[tid] += scratch[tid + stride];
                }
                __syncthreads();
            }
            if (tid == 0) {
                logit = scratch[0] * softmax_scale;
            }
            __syncthreads();
        }
        if (tid == 0) {
            logits[block] = logit;
            row_max = fmaxf(row_max, logit);
        }
        __syncthreads();
    }

    for (size_t j = 0; j < key_len; ++j) {
        const size_t k_offset = k_base + j * num_kv_heads * head_dim;
        float local_dot = 0.0f;
        for (size_t hd = tid; hd < head_dim; hd += blockDim.x) {
            local_dot += to_float<T>(q[q_base + hd]) * to_float<T>(k[k_offset + hd]);
        }
        scratch[tid] = local_dot;
        __syncthreads();
        for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                scratch[tid] += scratch[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            const float logit = scratch[0] * softmax_scale;
            logits[num_blocks + j] = logit;
            row_max = fmaxf(row_max, logit);
        }
        __syncthreads();
    }

    if (tid == 0) {
        float denom = expf(to_float<SinkT>(attn_sink[h]) - row_max);
        for (size_t idx = 0; idx < total_keys; ++idx) {
            denom += expf(logits[idx] - row_max);
        }
        scratch[0] = row_max;
        scratch[1] = denom;
    }
    __syncthreads();

    const float max_logit = scratch[0];
    const float denom = scratch[1];
    const size_t pass_dim = head_dim - rope_dim;

    for (size_t d = tid; d < pass_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (size_t block = 0; block < num_blocks; ++block) {
            if (block < visible_blocks && block_selected<IndexT>(indexed_blocks, index_top_k, query_len, b, tq, block)) {
                const float prob = expf(logits[block] - max_logit) / denom;
                const size_t comp_base = comp_batch_base + block * head_dim;
                acc += prob * to_float<T>(kv_comp[comp_base + d]);
            }
        }
        for (size_t j = 0; j < key_len; ++j) {
            const float prob = expf(logits[num_blocks + j] - max_logit) / denom;
            const size_t k_offset = k_base + j * num_kv_heads * head_dim;
            acc += prob * to_float<T>(k[k_offset + d]);
        }
        y[q_base + d] = from_float<T>(acc);
    }

    const size_t half = rope_dim / 2;
    for (size_t pair = tid; pair < half; pair += blockDim.x) {
        const size_t even = pass_dim + 2 * pair;
        const size_t odd = even + 1;
        float acc_even = 0.0f;
        float acc_odd = 0.0f;
        for (size_t block = 0; block < num_blocks; ++block) {
            if (block < visible_blocks && block_selected<IndexT>(indexed_blocks, index_top_k, query_len, b, tq, block)) {
                const float prob = expf(logits[block] - max_logit) / denom;
                const size_t comp_base = comp_batch_base + block * head_dim;
                const int64_t block_pos = read_pos<PosT>(block_positions, block);
                acc_even += prob * rotated_comp_value<T>(kv_comp, comp_base, even, head_dim,
                                                         rope_dim, block_pos, rope_theta,
                                                         use_yarn, yarn_factor, yarn_beta_fast,
                                                         yarn_beta_slow, yarn_original_seq_len,
                                                         yarn_extrapolation_factor);
                acc_odd += prob * rotated_comp_value<T>(kv_comp, comp_base, odd, head_dim,
                                                        rope_dim, block_pos, rope_theta,
                                                        use_yarn, yarn_factor, yarn_beta_fast,
                                                        yarn_beta_slow, yarn_original_seq_len,
                                                        yarn_extrapolation_factor);
            }
        }
        for (size_t j = 0; j < key_len; ++j) {
            const float prob = expf(logits[num_blocks + j] - max_logit) / denom;
            const size_t k_offset = k_base + j * num_kv_heads * head_dim;
            acc_even += prob * to_float<T>(k[k_offset + even]);
            acc_odd += prob * to_float<T>(k[k_offset + odd]);
        }
        const double inv_freq = rope_inv_freq(pair, rope_dim, rope_theta, use_yarn, yarn_factor,
                                              yarn_beta_fast, yarn_beta_slow,
                                              yarn_original_seq_len, yarn_extrapolation_factor);
        const double angle = static_cast<double>(q_pos) * inv_freq;
        const float c = static_cast<float>(cos(angle));
        const float s = static_cast<float>(-sin(angle));
        y[q_base + even] = from_float<T>(acc_even * c - acc_odd * s);
        y[q_base + odd] = from_float<T>(acc_odd * c + acc_even * s);
    }
}

template <typename T, typename SinkT, typename PosT, typename IndexT>
infiniStatus_t launch_typed_pos(const DeepseekV4CompressedDecodeInfo &info,
                                void *y,
                                const void *q,
                                const void *k,
                                const void *kv_comp,
                                const void *attn_sink,
                                const void *query_positions,
                                const void *block_positions,
                                const void *indexed_blocks,
                                cudaStream_t stream) {
    constexpr int threads = 256;
    const dim3 block(threads);
    const dim3 grid(info.batch_size * info.query_len * info.num_heads);
    const size_t shared_bytes = (info.num_blocks + info.key_len + threads) * sizeof(float);
    compressed_decode_kernel<T, SinkT, PosT, IndexT><<<grid, block, shared_bytes, stream>>>(
        reinterpret_cast<T *>(y),
        reinterpret_cast<const T *>(q),
        reinterpret_cast<const T *>(k),
        reinterpret_cast<const T *>(kv_comp),
        reinterpret_cast<const SinkT *>(attn_sink),
        query_positions,
        block_positions,
        indexed_blocks,
        info.batch_size,
        info.query_len,
        info.num_heads,
        info.key_len,
        info.num_kv_heads,
        info.num_blocks,
        info.head_dim,
        info.index_top_k,
        info.softmax_scale,
        info.compress_ratio,
        info.rope_dim,
        info.rope_theta,
        info.use_yarn,
        info.yarn_factor,
        info.yarn_beta_fast,
        info.yarn_beta_slow,
        info.yarn_original_seq_len,
        info.yarn_extrapolation_factor);
    CHECK_CUDA(cudaGetLastError());
    return INFINI_STATUS_SUCCESS;
}

template <typename T, typename SinkT>
infiniStatus_t launch_by_pos_dtype(const DeepseekV4CompressedDecodeInfo &info,
                                   void *y,
                                   const void *q,
                                   const void *k,
                                   const void *kv_comp,
                                   const void *attn_sink,
                                   const void *query_positions,
                                   const void *block_positions,
                                   const void *indexed_blocks,
                                   cudaStream_t stream) {
    if (info.positions_dtype == INFINI_DTYPE_I64) {
        if (info.indexed_dtype == INFINI_DTYPE_I64) {
            return launch_typed_pos<T, SinkT, int64_t, int64_t>(info, y, q, k, kv_comp, attn_sink,
                                                               query_positions, block_positions, indexed_blocks, stream);
        }
        if (info.indexed_dtype == INFINI_DTYPE_I32) {
            return launch_typed_pos<T, SinkT, int64_t, int32_t>(info, y, q, k, kv_comp, attn_sink,
                                                               query_positions, block_positions, indexed_blocks, stream);
        }
    }
    if (info.positions_dtype == INFINI_DTYPE_I32) {
        if (info.indexed_dtype == INFINI_DTYPE_I64) {
            return launch_typed_pos<T, SinkT, int32_t, int64_t>(info, y, q, k, kv_comp, attn_sink,
                                                               query_positions, block_positions, indexed_blocks, stream);
        }
        if (info.indexed_dtype == INFINI_DTYPE_I32) {
            return launch_typed_pos<T, SinkT, int32_t, int32_t>(info, y, q, k, kv_comp, attn_sink,
                                                               query_positions, block_positions, indexed_blocks, stream);
        }
    }
    return INFINI_STATUS_BAD_TENSOR_DTYPE;
}

template <typename T>
infiniStatus_t launch_by_sink_dtype(const DeepseekV4CompressedDecodeInfo &info,
                                    void *y,
                                    const void *q,
                                    const void *k,
                                    const void *kv_comp,
                                    const void *attn_sink,
                                    const void *query_positions,
                                    const void *block_positions,
                                    const void *indexed_blocks,
                                    cudaStream_t stream) {
    if (info.sink_dtype == INFINI_DTYPE_F16) {
        return launch_by_pos_dtype<T, half>(info, y, q, k, kv_comp, attn_sink,
                                           query_positions, block_positions, indexed_blocks, stream);
    }
    if (info.sink_dtype == INFINI_DTYPE_BF16) {
        return launch_by_pos_dtype<T, __nv_bfloat16>(info, y, q, k, kv_comp, attn_sink,
                                                     query_positions, block_positions, indexed_blocks, stream);
    }
    if (info.sink_dtype == INFINI_DTYPE_F32) {
        return launch_by_pos_dtype<T, float>(info, y, q, k, kv_comp, attn_sink,
                                             query_positions, block_positions, indexed_blocks, stream);
    }
    return INFINI_STATUS_BAD_TENSOR_DTYPE;
}

} // namespace

infiniStatus_t Descriptor::create(infiniopHandle_t handle,
                                  Descriptor **desc_ptr,
                                  infiniopTensorDescriptor_t y_desc,
                                  infiniopTensorDescriptor_t q_desc,
                                  infiniopTensorDescriptor_t k_desc,
                                  infiniopTensorDescriptor_t kv_comp_desc,
                                  infiniopTensorDescriptor_t attn_sink_desc,
                                  infiniopTensorDescriptor_t query_positions_desc,
                                  infiniopTensorDescriptor_t block_positions_desc,
                                  infiniopTensorDescriptor_t indexed_blocks_desc,
                                  float softmax_scale,
                                  size_t compress_ratio,
                                  size_t index_top_k,
                                  size_t rope_dim,
                                  double rope_theta,
                                  bool use_yarn,
                                  double yarn_factor,
                                  double yarn_beta_fast,
                                  double yarn_beta_slow,
                                  int64_t yarn_original_seq_len,
                                  double yarn_extrapolation_factor) {
    auto result = DeepseekV4CompressedDecodeInfo::create(y_desc, q_desc, k_desc, kv_comp_desc,
                                                         attn_sink_desc, query_positions_desc,
                                                         block_positions_desc, indexed_blocks_desc, softmax_scale,
                                                         compress_ratio, index_top_k, rope_dim, rope_theta,
                                                         use_yarn, yarn_factor, yarn_beta_fast,
                                                         yarn_beta_slow, yarn_original_seq_len,
                                                         yarn_extrapolation_factor);
    CHECK_RESULT(result);
    *desc_ptr = new Descriptor(result.take(), 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(void *workspace,
                                     size_t workspace_size,
                                     void *y,
                                     const void *q,
                                     const void *k,
                                     const void *kv_comp,
                                     const void *attn_sink,
                                     const void *query_positions,
                                     const void *block_positions,
                                     const void *indexed_blocks,
                                     void *stream) const {
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    if (_info.dtype == INFINI_DTYPE_F16) {
        return launch_by_sink_dtype<half>(_info, y, q, k, kv_comp, attn_sink,
                                          query_positions, block_positions, indexed_blocks, (cudaStream_t)stream);
    }
    if (_info.dtype == INFINI_DTYPE_BF16) {
        return launch_by_sink_dtype<__nv_bfloat16>(_info, y, q, k, kv_comp, attn_sink,
                                                   query_positions, block_positions, indexed_blocks, (cudaStream_t)stream);
    }
    if (_info.dtype == INFINI_DTYPE_F32) {
        return launch_by_sink_dtype<float>(_info, y, q, k, kv_comp, attn_sink,
                                           query_positions, block_positions, indexed_blocks, (cudaStream_t)stream);
    }
    return INFINI_STATUS_BAD_TENSOR_DTYPE;
}

} // namespace op::deepseek_v4_compressed_decode::nvidia

#endif
