#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_HYGON_API) || defined(ENABLE_ALI_API)

#include "deepseek_v4_compressor_nvidia.cuh"

#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <memory>
#include <type_traits>

namespace op::deepseek_v4_compressor::nvidia {

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

template <typename T>
__device__ T from_float(float value) {
    if constexpr (std::is_same_v<T, half>) {
        return __float2half(value);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return __float2bfloat16(value);
    } else {
        return value;
    }
}

template <typename T>
__device__ float round_to_dtype(float value) {
    return to_float<T>(from_float<T>(value));
}

template <typename T>
__device__ void load_entry(
    const T *__restrict__ kv,
    const T *__restrict__ score,
    const T *__restrict__ ape,
    size_t b,
    size_t block,
    size_t d,
    size_t entry,
    size_t seq_len,
    size_t compressed_dim,
    size_t compress_ratio,
    size_t head_dim,
    size_t coff,
    float &value,
    float &score_value) {
    value = 0.0f;
    score_value = -CUDART_INF_F;
    if (coff == 1) {
        const size_t r = entry;
        const size_t src = block * compress_ratio + r;
        if (r < compress_ratio && src < seq_len) {
            const size_t offset = (b * seq_len + src) * compressed_dim + d;
            value = to_float<T>(kv[offset]);
            score_value = to_float<T>(score[offset]) + to_float<T>(ape[r * compressed_dim + d]);
        }
        return;
    }

    if (entry < compress_ratio) {
        if (block == 0) {
            return;
        }
        const size_t r = entry;
        const size_t src = (block - 1) * compress_ratio + r;
        if (src < seq_len) {
            const size_t offset = (b * seq_len + src) * compressed_dim + d;
            value = to_float<T>(kv[offset]);
            score_value = to_float<T>(score[offset]) + to_float<T>(ape[r * compressed_dim + d]);
        }
    } else {
        const size_t r = entry - compress_ratio;
        const size_t src = block * compress_ratio + r;
        if (r < compress_ratio && src < seq_len) {
            const size_t col = head_dim + d;
            const size_t offset = (b * seq_len + src) * compressed_dim + col;
            value = to_float<T>(kv[offset]);
            score_value = to_float<T>(score[offset]) + to_float<T>(ape[r * compressed_dim + col]);
        }
    }
}

template <typename T, int THREADS>
__global__ void deepseek_v4_compressor_kernel(
    const T *__restrict__ kv,
    const T *__restrict__ score,
    const T *__restrict__ ape,
    const T *__restrict__ norm_weight,
    T *__restrict__ out,
    size_t batch_size,
    size_t seq_len,
    size_t num_blocks,
    size_t head_dim,
    size_t compressed_dim,
    size_t compress_ratio,
    size_t coff,
    float epsilon) {
    const size_t row = blockIdx.x;
    if (row >= batch_size * num_blocks) {
        return;
    }
    const size_t b = row / num_blocks;
    const size_t block = row - b * num_blocks;
    const int tid = threadIdx.x;
    extern __shared__ float pooled[];
    __shared__ float reduce_sums[THREADS];

    float local_sum = 0.0f;
    const size_t pool_len = coff == 2 ? 2 * compress_ratio : compress_ratio;
    for (size_t d = static_cast<size_t>(tid); d < head_dim; d += THREADS) {
        float max_score = -CUDART_INF_F;
        for (size_t entry = 0; entry < pool_len; ++entry) {
            float value = 0.0f;
            float score_value = -CUDART_INF_F;
            load_entry<T>(kv, score, ape, b, block, d, entry, seq_len, compressed_dim,
                          compress_ratio, head_dim, coff, value, score_value);
            max_score = fmaxf(max_score, score_value);
        }

        float denom = 0.0f;
        float acc = 0.0f;
        for (size_t entry = 0; entry < pool_len; ++entry) {
            float value = 0.0f;
            float score_value = -CUDART_INF_F;
            load_entry<T>(kv, score, ape, b, block, d, entry, seq_len, compressed_dim,
                          compress_ratio, head_dim, coff, value, score_value);
            if (isfinite(score_value)) {
                const float w = expf(score_value - max_score);
                denom += w;
                acc += w * value;
            }
        }
        float pooled_value = denom > 0.0f ? acc / denom : 0.0f;
        pooled_value = round_to_dtype<T>(pooled_value);
        pooled[d] = pooled_value;
        local_sum += pooled_value * pooled_value;
    }

    reduce_sums[tid] = local_sum;
    __syncthreads();
    for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduce_sums[tid] += reduce_sums[tid + stride];
        }
        __syncthreads();
    }
    const float inv_rms = rsqrtf(reduce_sums[0] / static_cast<float>(head_dim) + epsilon);
    __syncthreads();

    for (size_t d = static_cast<size_t>(tid); d < head_dim; d += THREADS) {
        const float normalized = pooled[d] * inv_rms * to_float<T>(norm_weight[d]);
        out[(b * num_blocks + block) * head_dim + d] = from_float<T>(normalized);
    }
}

template <typename T>
infiniStatus_t launch_typed(
    const DeepseekV4CompressorInfo &info,
    void *out,
    const void *kv,
    const void *score,
    const void *ape,
    const void *norm_weight,
    cudaStream_t stream) {
    constexpr int THREADS = 256;
    const size_t rows = info.batch_size * info.num_blocks;
    const size_t shared_bytes = info.head_dim * sizeof(float);
    deepseek_v4_compressor_kernel<T, THREADS><<<rows, THREADS, shared_bytes, stream>>>(
        reinterpret_cast<const T *>(kv),
        reinterpret_cast<const T *>(score),
        reinterpret_cast<const T *>(ape),
        reinterpret_cast<const T *>(norm_weight),
        reinterpret_cast<T *>(out),
        info.batch_size,
        info.seq_len,
        info.num_blocks,
        info.head_dim,
        info.compressed_dim,
        info.compress_ratio,
        info.coff,
        info.epsilon);
    CHECK_CUDA(cudaGetLastError());
    return INFINI_STATUS_SUCCESS;
}

} // namespace

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t kv_desc,
    infiniopTensorDescriptor_t score_desc,
    infiniopTensorDescriptor_t ape_desc,
    infiniopTensorDescriptor_t norm_weight_desc,
    size_t compress_ratio,
    float epsilon) {
    auto result = DeepseekV4CompressorInfo::create(
        out_desc, kv_desc, score_desc, ape_desc, norm_weight_desc, compress_ratio, epsilon);
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
    void *out,
    const void *kv,
    const void *score,
    const void *ape,
    const void *norm_weight,
    void *stream_) const {
    (void)workspace;
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    auto stream = reinterpret_cast<cudaStream_t>(stream_);
    if (_info.dtype == INFINI_DTYPE_F16) {
        return launch_typed<half>(_info, out, kv, score, ape, norm_weight, stream);
    }
    if (_info.dtype == INFINI_DTYPE_BF16) {
        return launch_typed<__nv_bfloat16>(_info, out, kv, score, ape, norm_weight, stream);
    }
    if (_info.dtype == INFINI_DTYPE_F32) {
        return launch_typed<float>(_info, out, kv, score, ape, norm_weight, stream);
    }
    return INFINI_STATUS_BAD_TENSOR_DTYPE;
}

} // namespace op::deepseek_v4_compressor::nvidia

#endif
