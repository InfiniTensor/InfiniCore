#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_HYGON_API) || defined(ENABLE_ALI_API)

#include "deepseek_v4_swa_decode_nvidia.cuh"

#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <type_traits>

namespace op::deepseek_v4_swa_decode::nvidia {
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

template <typename T, typename SinkT>
__global__ void swa_decode_kernel(T *__restrict__ y,
                                  const T *__restrict__ q,
                                  const T *__restrict__ k,
                                  const SinkT *__restrict__ attn_sink,
                                  size_t batch_size,
                                  size_t query_len,
                                  size_t num_heads,
                                  size_t key_len,
                                  size_t num_kv_heads,
                                  size_t head_dim,
                                  float softmax_scale) {
    extern __shared__ float smem[];
    float *logits = smem;
    float *scratch = smem + key_len;

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

    const size_t q_base = ((b * query_len + tq) * num_heads + h) * head_dim;
    const size_t k_base = (b * key_len * num_kv_heads + kv_head) * head_dim;

    float row_max = to_float<SinkT>(attn_sink[h]);
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
            logits[j] = logit;
            row_max = fmaxf(row_max, logit);
        }
        __syncthreads();
    }

    if (tid == 0) {
        float denom = expf(to_float<SinkT>(attn_sink[h]) - row_max);
        for (size_t j = 0; j < key_len; ++j) {
            denom += expf(logits[j] - row_max);
        }
        scratch[0] = row_max;
        scratch[1] = denom;
    }
    __syncthreads();

    const float max_logit = scratch[0];
    const float denom = scratch[1];
    for (size_t d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (size_t j = 0; j < key_len; ++j) {
            const size_t k_offset = k_base + j * num_kv_heads * head_dim;
            const float prob = expf(logits[j] - max_logit) / denom;
            acc += prob * to_float<T>(k[k_offset + d]);
        }
        y[q_base + d] = from_float<T>(acc);
    }
}

template <typename T, typename SinkT>
infiniStatus_t launch_typed(const DeepseekV4SwaDecodeInfo &info,
                            void *y,
                            const void *q,
                            const void *k,
                            const void *attn_sink,
                            cudaStream_t stream) {
    constexpr int threads = 256;
    const dim3 block(threads);
    const dim3 grid(info.batch_size * info.query_len * info.num_heads);
    const size_t shared_bytes = (info.key_len + threads) * sizeof(float);
    swa_decode_kernel<T, SinkT><<<grid, block, shared_bytes, stream>>>(
        reinterpret_cast<T *>(y),
        reinterpret_cast<const T *>(q),
        reinterpret_cast<const T *>(k),
        reinterpret_cast<const SinkT *>(attn_sink),
        info.batch_size,
        info.query_len,
        info.num_heads,
        info.key_len,
        info.num_kv_heads,
        info.head_dim,
        info.softmax_scale);
    CHECK_CUDA(cudaGetLastError());
    return INFINI_STATUS_SUCCESS;
}

template <typename T>
infiniStatus_t launch_by_sink_dtype(const DeepseekV4SwaDecodeInfo &info,
                                    void *y,
                                    const void *q,
                                    const void *k,
                                    const void *attn_sink,
                                    cudaStream_t stream) {
    if (info.sink_dtype == INFINI_DTYPE_F16) {
        return launch_typed<T, half>(info, y, q, k, attn_sink, stream);
    }
    if (info.sink_dtype == INFINI_DTYPE_BF16) {
        return launch_typed<T, __nv_bfloat16>(info, y, q, k, attn_sink, stream);
    }
    if (info.sink_dtype == INFINI_DTYPE_F32) {
        return launch_typed<T, float>(info, y, q, k, attn_sink, stream);
    }
    return INFINI_STATUS_BAD_TENSOR_DTYPE;
}

} // namespace

infiniStatus_t Descriptor::create(infiniopHandle_t handle,
                                  Descriptor **desc_ptr,
                                  infiniopTensorDescriptor_t y_desc,
                                  infiniopTensorDescriptor_t q_desc,
                                  infiniopTensorDescriptor_t k_desc,
                                  infiniopTensorDescriptor_t attn_sink_desc,
                                  float softmax_scale) {
    auto result = DeepseekV4SwaDecodeInfo::create(y_desc, q_desc, k_desc, attn_sink_desc, softmax_scale);
    CHECK_RESULT(result);
    *desc_ptr = new Descriptor(result.take(), 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(void *workspace,
                                     size_t workspace_size,
                                     void *y,
                                     const void *q,
                                     const void *k,
                                     const void *attn_sink,
                                     void *stream) const {
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    if (_info.dtype == INFINI_DTYPE_F16) {
        return launch_by_sink_dtype<half>(_info, y, q, k, attn_sink, (cudaStream_t)stream);
    }
    if (_info.dtype == INFINI_DTYPE_BF16) {
        return launch_by_sink_dtype<__nv_bfloat16>(_info, y, q, k, attn_sink, (cudaStream_t)stream);
    }
    if (_info.dtype == INFINI_DTYPE_F32) {
        return launch_by_sink_dtype<float>(_info, y, q, k, attn_sink, (cudaStream_t)stream);
    }
    return INFINI_STATUS_BAD_TENSOR_DTYPE;
}

} // namespace op::deepseek_v4_swa_decode::nvidia

#endif
