#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_HYGON_API) || defined(ENABLE_ALI_API)

#include "deepseek_v4_router_nvidia.cuh"

#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <memory>
#include <type_traits>

namespace op::deepseek_v4_router::nvidia {

struct TopkRouterDescriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

struct HashRouterDescriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

struct HashTopkRouterDescriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

TopkRouterDescriptor::~TopkRouterDescriptor() {
    delete _opaque;
}

HashRouterDescriptor::~HashRouterDescriptor() {
    delete _opaque;
}

HashTopkRouterDescriptor::~HashTopkRouterDescriptor() {
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

__device__ float sqrtsoftplus(float x) {
    return sqrtf(log1pf(expf(-fabsf(x))) + fmaxf(x, 0.0f));
}

__device__ bool better_pair(float candidate_score, int candidate_expert, float best_score, int best_expert) {
    return candidate_score > best_score || (candidate_score == best_score && candidate_expert < best_expert);
}

template <typename T>
__device__ void reduce_pair_warp(float &score, int &expert) {
    constexpr unsigned FULL_MASK = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other_score = __shfl_down_sync(FULL_MASK, score, offset);
        const int other_expert = __shfl_down_sync(FULL_MASK, expert, offset);
        if (better_pair(other_score, other_expert, score, expert)) {
            score = other_score;
            expert = other_expert;
        }
    }
}

template <typename T>
__global__ void topk_router_kernel(
    const T *__restrict__ logits,
    const float *__restrict__ bias,
    float *__restrict__ topk_weights,
    int *__restrict__ topk_indices,
    size_t num_tokens,
    size_t num_experts,
    size_t topk,
    bool renormalize) {
    constexpr int THREADS = 256;
    constexpr int WARP_SIZE = 32;
    constexpr int WARPS = THREADS / WARP_SIZE;
    const size_t token = blockIdx.x;
    if (token >= num_tokens) {
        return;
    }

    __shared__ float raw_scores[THREADS];
    __shared__ float select_scores[THREADS];
    __shared__ float warp_scores[WARPS];
    __shared__ int warp_experts[WARPS];
    __shared__ int selected_experts[32];
    __shared__ float selected_weights[32];

    const int tid = threadIdx.x;
    const int lane = tid & (WARP_SIZE - 1);
    const int warp = tid / WARP_SIZE;
    const bool active = static_cast<size_t>(tid) < num_experts;

    float raw = -CUDART_INF_F;
    float select = -CUDART_INF_F;
    if (active) {
        raw = sqrtsoftplus(to_float<T>(logits[token * num_experts + tid]));
        select = raw + (bias == nullptr ? 0.0f : bias[tid]);
    }
    raw_scores[tid] = raw;
    select_scores[tid] = select;
    __syncthreads();

    float denom = 0.0f;
    for (size_t k = 0; k < topk; ++k) {
        float best_score = active ? select_scores[tid] : -CUDART_INF_F;
        int best_expert = active ? tid : static_cast<int>(num_experts);
        reduce_pair_warp<T>(best_score, best_expert);

        if (lane == 0) {
            warp_scores[warp] = best_score;
            warp_experts[warp] = best_expert;
        }
        __syncthreads();

        if (warp == 0) {
            best_score = lane < WARPS ? warp_scores[lane] : -CUDART_INF_F;
            best_expert = lane < WARPS ? warp_experts[lane] : static_cast<int>(num_experts);
            reduce_pair_warp<T>(best_score, best_expert);
            if (lane == 0) {
                selected_experts[k] = best_expert;
                selected_weights[k] = raw_scores[best_expert];
                denom += selected_weights[k];
            }
        }
        __syncthreads();

        const int winner = selected_experts[k];
        if (active && tid == winner) {
            select_scores[tid] = -CUDART_INF_F;
        }
        __syncthreads();
    }

    if (tid == 0) {
        const float inv = renormalize ? 1.0f / (denom + 1.0e-9f) : 1.0f;
        const size_t offset = token * topk;
        for (size_t k = 0; k < topk; ++k) {
            topk_indices[offset + k] = selected_experts[k];
            topk_weights[offset + k] = selected_weights[k] * inv;
        }
    }
}

template <typename IdT>
__device__ long long load_id(const void *ptr, size_t idx) {
    return static_cast<long long>(reinterpret_cast<const IdT *>(ptr)[idx]);
}

__device__ long long load_id_by_dtype(const void *ptr, size_t idx, infiniDtype_t dtype) {
    if (dtype == INFINI_DTYPE_I32) {
        return load_id<int>(ptr, idx);
    }
    return load_id<long long>(ptr, idx);
}


template <typename T>
__global__ void hash_router_warp_kernel(
    const T *__restrict__ logits,
    const void *__restrict__ input_ids,
    const void *__restrict__ tid2eid,
    float *__restrict__ topk_weights,
    int *__restrict__ topk_indices,
    size_t num_tokens,
    size_t num_experts,
    size_t topk,
    size_t vocab_size,
    infiniDtype_t input_ids_dtype,
    infiniDtype_t tid2eid_dtype,
    bool renormalize) {
    constexpr int WARP_SIZE = 32;
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;
    const size_t token = static_cast<size_t>(blockIdx.x) * warps_per_block + warp;
    if (token >= num_tokens) {
        return;
    }

    const long long token_id = load_id_by_dtype(input_ids, token, input_ids_dtype);
    float weight = 0.0f;
    int expert_i32 = 0;
    if (lane < static_cast<int>(topk) && token_id >= 0 && static_cast<size_t>(token_id) < vocab_size) {
        const long long expert_id = load_id_by_dtype(tid2eid, static_cast<size_t>(token_id) * topk + lane, tid2eid_dtype);
        if (expert_id >= 0 && static_cast<size_t>(expert_id) < num_experts) {
            expert_i32 = static_cast<int>(expert_id);
            weight = sqrtsoftplus(to_float<T>(logits[token * num_experts + static_cast<size_t>(expert_id)]));
        }
    }

    float denom = weight;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        denom += __shfl_down_sync(0xffffffff, denom, offset);
    }
    denom = __shfl_sync(0xffffffff, denom, 0);

    if (lane < static_cast<int>(topk)) {
        const size_t offset = token * topk + lane;
        topk_indices[offset] = expert_i32;
        const float inv = renormalize ? 1.0f / (denom + 1.0e-9f) : 1.0f;
        topk_weights[offset] = weight * inv;
    }
}

template <typename T>
__global__ void hash_router_kernel(
    const T *logits,
    const void *input_ids,
    const void *tid2eid,
    float *topk_weights,
    int *topk_indices,
    size_t num_tokens,
    size_t num_experts,
    size_t topk,
    size_t vocab_size,
    infiniDtype_t input_ids_dtype,
    infiniDtype_t tid2eid_dtype,
    bool renormalize) {
    const size_t token = blockIdx.x;
    if (token >= num_tokens) {
        return;
    }

    const int tid = threadIdx.x;
    extern __shared__ float shared_weights[];
    const long long token_id = load_id_by_dtype(input_ids, token, input_ids_dtype);
    if (token_id < 0 || static_cast<size_t>(token_id) >= vocab_size) {
        if (tid == 0) {
            const size_t offset = token * topk;
            for (size_t k = 0; k < topk; ++k) {
                topk_indices[offset + k] = 0;
                topk_weights[offset + k] = 0.0f;
            }
        }
        return;
    }

    float denom = 0.0f;
    for (size_t k = tid; k < topk; k += blockDim.x) {
        const long long expert_id = load_id_by_dtype(tid2eid, static_cast<size_t>(token_id) * topk + k, tid2eid_dtype);
        float weight = 0.0f;
        int expert_i32 = 0;
        if (expert_id >= 0 && static_cast<size_t>(expert_id) < num_experts) {
            expert_i32 = static_cast<int>(expert_id);
            weight = sqrtsoftplus(to_float<T>(logits[token * num_experts + static_cast<size_t>(expert_id)]));
        }
        topk_indices[token * topk + k] = expert_i32;
        shared_weights[k] = weight;
    }
    __syncthreads();

    if (tid == 0) {
        for (size_t k = 0; k < topk; ++k) {
            denom += shared_weights[k];
        }
        const float inv = renormalize ? 1.0f / (denom + 1.0e-9f) : 1.0f;
        const size_t offset = token * topk;
        for (size_t k = 0; k < topk; ++k) {
            topk_weights[offset + k] = shared_weights[k] * inv;
        }
    }
}


template <typename T>
__global__ void hash_topk_selected_linear_kernel(
    const T *__restrict__ hidden_states,
    const T *__restrict__ weight,
    const void *__restrict__ input_ids,
    const void *__restrict__ tid2eid,
    float *__restrict__ topk_weights,
    int *__restrict__ topk_indices,
    size_t num_tokens,
    size_t hidden_size,
    size_t num_experts,
    size_t topk,
    size_t vocab_size,
    infiniDtype_t input_ids_dtype,
    infiniDtype_t tid2eid_dtype,
    bool renormalize) {
    constexpr int MAX_TOPK = 16;
    const size_t token = blockIdx.x;
    const int tid = threadIdx.x;
    if (token >= num_tokens || topk > MAX_TOPK) {
        return;
    }

    extern __shared__ float shared[];
    float *partials = shared;
    float *scores = shared + MAX_TOPK * blockDim.x;
    int *experts = reinterpret_cast<int *>(scores + MAX_TOPK);

    float local[MAX_TOPK];
    #pragma unroll
    for (int k = 0; k < MAX_TOPK; ++k) {
        local[k] = 0.0f;
    }

    const long long token_id = load_id_by_dtype(input_ids, token, input_ids_dtype);
    if (tid < static_cast<int>(topk)) {
        int expert_i32 = 0;
        if (token_id >= 0 && static_cast<size_t>(token_id) < vocab_size) {
            const long long expert_id = load_id_by_dtype(tid2eid, static_cast<size_t>(token_id) * topk + tid, tid2eid_dtype);
            if (expert_id >= 0 && static_cast<size_t>(expert_id) < num_experts) {
                expert_i32 = static_cast<int>(expert_id);
            }
        }
        experts[tid] = expert_i32;
        topk_indices[token * topk + tid] = expert_i32;
    }
    __syncthreads();

    const T *hidden = hidden_states + token * hidden_size;
    for (size_t h = tid; h < hidden_size; h += blockDim.x) {
        const float hv = to_float<T>(hidden[h]);
        #pragma unroll
        for (int k = 0; k < MAX_TOPK; ++k) {
            if (k < static_cast<int>(topk)) {
                const int expert = experts[k];
                local[k] += hv * to_float<T>(weight[static_cast<size_t>(expert) * hidden_size + h]);
            }
        }
    }

    #pragma unroll
    for (int k = 0; k < MAX_TOPK; ++k) {
        if (k < static_cast<int>(topk)) {
            partials[k * blockDim.x + tid] = local[k];
        }
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            #pragma unroll
            for (int k = 0; k < MAX_TOPK; ++k) {
                if (k < static_cast<int>(topk)) {
                    partials[k * blockDim.x + tid] += partials[k * blockDim.x + tid + stride];
                }
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        float denom = 0.0f;
        #pragma unroll
        for (int k = 0; k < MAX_TOPK; ++k) {
            if (k < static_cast<int>(topk)) {
                const float score = sqrtsoftplus(partials[k * blockDim.x]);
                scores[k] = score;
                denom += score;
            }
        }
        const float inv = renormalize ? 1.0f / (denom + 1.0e-9f) : 1.0f;
        for (size_t k = 0; k < topk; ++k) {
            topk_weights[token * topk + k] = scores[k] * inv;
        }
    }
}

template <typename T>
infiniStatus_t launch_hash_topk_typed(
    const DeepseekV4HashTopkRouterInfo &info,
    float *topk_weights,
    int *topk_indices,
    const void *hidden_states,
    const void *weight,
    const void *input_ids,
    const void *tid2eid,
    cudaStream_t stream) {
    if (info.topk > 16) {
        return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
    }
    constexpr int threads = 256;
    const size_t shared_bytes = (16 * threads + 16) * sizeof(float) + 16 * sizeof(int);
    hash_topk_selected_linear_kernel<T><<<info.num_tokens, threads, shared_bytes, stream>>>(
        reinterpret_cast<const T *>(hidden_states),
        reinterpret_cast<const T *>(weight),
        input_ids,
        tid2eid,
        topk_weights,
        topk_indices,
        info.num_tokens,
        info.hidden_size,
        info.num_experts,
        info.topk,
        info.vocab_size,
        info.input_ids_dtype,
        info.tid2eid_dtype,
        info.renormalize);
    CHECK_CUDA(cudaGetLastError());
    return INFINI_STATUS_SUCCESS;
}

template <typename T>
infiniStatus_t launch_topk_typed(
    const DeepseekV4TopkRouterInfo &info,
    float *topk_weights,
    int *topk_indices,
    const void *logits,
    const void *bias,
    cudaStream_t stream) {
    if (info.num_experts > 256 || info.topk > 32) {
        return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
    }
    topk_router_kernel<T><<<info.num_tokens, 256, 0, stream>>>(
        reinterpret_cast<const T *>(logits),
        reinterpret_cast<const float *>(bias),
        topk_weights,
        topk_indices,
        info.num_tokens,
        info.num_experts,
        info.topk,
        info.renormalize);
    CHECK_CUDA(cudaGetLastError());
    return INFINI_STATUS_SUCCESS;
}

template <typename T>
infiniStatus_t launch_hash_typed(
    const DeepseekV4HashRouterInfo &info,
    float *topk_weights,
    int *topk_indices,
    const void *logits,
    const void *input_ids,
    const void *tid2eid,
    cudaStream_t stream) {
    if (info.topk > 256) {
        return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
    }
    if (info.topk <= 32) {
        constexpr int threads = 128;
        constexpr int warps_per_block = threads / 32;
        const dim3 grid((info.num_tokens + warps_per_block - 1) / warps_per_block);
        hash_router_warp_kernel<T><<<grid, threads, 0, stream>>>(
            reinterpret_cast<const T *>(logits),
            input_ids,
            tid2eid,
            topk_weights,
            topk_indices,
            info.num_tokens,
            info.num_experts,
            info.topk,
            info.vocab_size,
            info.input_ids_dtype,
            info.tid2eid_dtype,
            info.renormalize);
    } else {
        const int threads = 256;
        const size_t shared_bytes = info.topk * sizeof(float);
        hash_router_kernel<T><<<info.num_tokens, threads, shared_bytes, stream>>>(
            reinterpret_cast<const T *>(logits),
            input_ids,
            tid2eid,
            topk_weights,
            topk_indices,
            info.num_tokens,
            info.num_experts,
            info.topk,
            info.vocab_size,
            info.input_ids_dtype,
            info.tid2eid_dtype,
            info.renormalize);
    }
    CHECK_CUDA(cudaGetLastError());
    return INFINI_STATUS_SUCCESS;
}

} // namespace

infiniStatus_t TopkRouterDescriptor::create(
    infiniopHandle_t handle,
    TopkRouterDescriptor **desc_ptr,
    infiniopTensorDescriptor_t topk_weights_desc,
    infiniopTensorDescriptor_t topk_indices_desc,
    infiniopTensorDescriptor_t logits_desc,
    infiniopTensorDescriptor_t bias_desc,
    bool renormalize) {
    auto result = DeepseekV4TopkRouterInfo::create(topk_weights_desc, topk_indices_desc, logits_desc, bias_desc, renormalize);
    CHECK_RESULT(result);
    auto info = result.take();
    *desc_ptr = new TopkRouterDescriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        info,
        0,
        handle->device,
        handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t TopkRouterDescriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *topk_weights,
    void *topk_indices,
    const void *logits,
    const void *bias,
    void *stream_) const {
    (void)workspace;
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    auto stream = reinterpret_cast<cudaStream_t>(stream_);
    if (_info.dtype == INFINI_DTYPE_F16) {
        return launch_topk_typed<half>(_info, reinterpret_cast<float *>(topk_weights), reinterpret_cast<int *>(topk_indices), logits, bias, stream);
    }
    if (_info.dtype == INFINI_DTYPE_BF16) {
        return launch_topk_typed<__nv_bfloat16>(_info, reinterpret_cast<float *>(topk_weights), reinterpret_cast<int *>(topk_indices), logits, bias, stream);
    }
    if (_info.dtype == INFINI_DTYPE_F32) {
        return launch_topk_typed<float>(_info, reinterpret_cast<float *>(topk_weights), reinterpret_cast<int *>(topk_indices), logits, bias, stream);
    }
    return INFINI_STATUS_BAD_TENSOR_DTYPE;
}

infiniStatus_t HashRouterDescriptor::create(
    infiniopHandle_t handle,
    HashRouterDescriptor **desc_ptr,
    infiniopTensorDescriptor_t topk_weights_desc,
    infiniopTensorDescriptor_t topk_indices_desc,
    infiniopTensorDescriptor_t logits_desc,
    infiniopTensorDescriptor_t input_ids_desc,
    infiniopTensorDescriptor_t tid2eid_desc,
    bool renormalize) {
    auto result = DeepseekV4HashRouterInfo::create(topk_weights_desc, topk_indices_desc, logits_desc, input_ids_desc, tid2eid_desc, renormalize);
    CHECK_RESULT(result);
    auto info = result.take();
    *desc_ptr = new HashRouterDescriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        info,
        0,
        handle->device,
        handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t HashRouterDescriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *topk_weights,
    void *topk_indices,
    const void *logits,
    const void *input_ids,
    const void *tid2eid,
    void *stream_) const {
    (void)workspace;
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    auto stream = reinterpret_cast<cudaStream_t>(stream_);
    if (_info.logits_dtype == INFINI_DTYPE_F16) {
        return launch_hash_typed<half>(_info, reinterpret_cast<float *>(topk_weights), reinterpret_cast<int *>(topk_indices), logits, input_ids, tid2eid, stream);
    }
    if (_info.logits_dtype == INFINI_DTYPE_BF16) {
        return launch_hash_typed<__nv_bfloat16>(_info, reinterpret_cast<float *>(topk_weights), reinterpret_cast<int *>(topk_indices), logits, input_ids, tid2eid, stream);
    }
    if (_info.logits_dtype == INFINI_DTYPE_F32) {
        return launch_hash_typed<float>(_info, reinterpret_cast<float *>(topk_weights), reinterpret_cast<int *>(topk_indices), logits, input_ids, tid2eid, stream);
    }
    return INFINI_STATUS_BAD_TENSOR_DTYPE;
}


infiniStatus_t HashTopkRouterDescriptor::create(
    infiniopHandle_t handle,
    HashTopkRouterDescriptor **desc_ptr,
    infiniopTensorDescriptor_t topk_weights_desc,
    infiniopTensorDescriptor_t topk_indices_desc,
    infiniopTensorDescriptor_t hidden_states_desc,
    infiniopTensorDescriptor_t weight_desc,
    infiniopTensorDescriptor_t input_ids_desc,
    infiniopTensorDescriptor_t tid2eid_desc,
    bool renormalize) {
    auto result = DeepseekV4HashTopkRouterInfo::create(topk_weights_desc, topk_indices_desc, hidden_states_desc, weight_desc, input_ids_desc, tid2eid_desc, renormalize);
    CHECK_RESULT(result);
    auto info = result.take();
    *desc_ptr = new HashTopkRouterDescriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        info,
        0,
        handle->device,
        handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t HashTopkRouterDescriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *topk_weights,
    void *topk_indices,
    const void *hidden_states,
    const void *weight,
    const void *input_ids,
    const void *tid2eid,
    void *stream_) const {
    (void)workspace;
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    auto stream = reinterpret_cast<cudaStream_t>(stream_);
    if (_info.dtype == INFINI_DTYPE_F16) {
        return launch_hash_topk_typed<half>(_info, reinterpret_cast<float *>(topk_weights), reinterpret_cast<int *>(topk_indices), hidden_states, weight, input_ids, tid2eid, stream);
    }
    if (_info.dtype == INFINI_DTYPE_BF16) {
        return launch_hash_topk_typed<__nv_bfloat16>(_info, reinterpret_cast<float *>(topk_weights), reinterpret_cast<int *>(topk_indices), hidden_states, weight, input_ids, tid2eid, stream);
    }
    if (_info.dtype == INFINI_DTYPE_F32) {
        return launch_hash_topk_typed<float>(_info, reinterpret_cast<float *>(topk_weights), reinterpret_cast<int *>(topk_indices), hidden_states, weight, input_ids, tid2eid, stream);
    }
    return INFINI_STATUS_BAD_TENSOR_DTYPE;
}

} // namespace op::deepseek_v4_router::nvidia

#endif
