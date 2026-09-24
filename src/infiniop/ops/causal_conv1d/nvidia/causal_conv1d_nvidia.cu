// causal_conv1d_nvidia.cu

#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "causal_conv1d_nvidia.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace op {
namespace causal_conv1d {
namespace nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t conv_state_desc,
    infiniopTensorDescriptor_t final_conv_state_desc,
    infiniopTensorDescriptor_t qkv_desc,
    infiniopTensorDescriptor_t weight_desc,
    infiniopTensorDescriptor_t bias_desc,
    infiniopTensorDescriptor_t cu_seqlens_desc,
    infiniopTensorDescriptor_t initial_state_indices_desc,
    infiniopTensorDescriptor_t final_state_indices_desc) {

    auto result = CausalConv1dInfo::create(
        out_desc, conv_state_desc, final_conv_state_desc, qkv_desc, weight_desc,
        bias_desc, cu_seqlens_desc, initial_state_indices_desc, final_state_indices_desc);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        result.take(),
        0,
        handle->device,
        handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

namespace {

template <typename T>
__device__ __forceinline__ float load_as_float(const T *ptr, ptrdiff_t offset) {
    return static_cast<float>(ptr[offset]);
}

template <>
__device__ __forceinline__ float load_as_float<half>(const half *ptr, ptrdiff_t offset) {
    return __half2float(ptr[offset]);
}

template <>
__device__ __forceinline__ float load_as_float<__nv_bfloat16>(const __nv_bfloat16 *ptr, ptrdiff_t offset) {
    return __bfloat162float(ptr[offset]);
}

template <typename T>
__device__ __forceinline__ T cast_from_float(float x) {
    return static_cast<T>(x);
}

template <>
__device__ __forceinline__ half cast_from_float<half>(float x) {
    return __float2half(x);
}

template <>
__device__ __forceinline__ __nv_bfloat16 cast_from_float<__nv_bfloat16>(float x) {
    return __float2bfloat16(x);
}

__device__ __forceinline__ int64_t load_optional_index(
    const void *indices,
    bool is_i64,
    int idx,
    int fallback) {
    if (indices == nullptr) {
        return static_cast<int64_t>(fallback);
    }
    return is_i64
             ? static_cast<const int64_t *>(indices)[idx]
             : static_cast<int64_t>(static_cast<const int32_t *>(indices)[idx]);
}

template <typename T>
__device__ __forceinline__ float load_history_k4(
    const T *conv_state,
    const T *qkv,
    int64_t history_pos,
    int64_t token_begin,
    int token_batch,
    int channel,
    ptrdiff_t state_base,
    ptrdiff_t state_s2,
    ptrdiff_t qkv_s0,
    ptrdiff_t qkv_s1,
    ptrdiff_t qkv_s2) {
    constexpr int64_t STATE_LEN = 3;
    if (history_pos < STATE_LEN) {
        return load_as_float(conv_state, state_base + static_cast<ptrdiff_t>(history_pos) * state_s2);
    }

    const int64_t token_idx = token_begin + history_pos - STATE_LEN;
    const ptrdiff_t qkv_off = static_cast<ptrdiff_t>(token_batch) * qkv_s0 + static_cast<ptrdiff_t>(token_idx) * qkv_s1 + static_cast<ptrdiff_t>(channel) * qkv_s2;
    return load_as_float(qkv, qkv_off);
}

template <typename T>
__global__ void causal_conv1d_k4_prefill_kernel(
    T *out,
    T *conv_state,
    T *final_conv_state,
    const T *qkv,
    const T *weight,
    const T *bias,
    const void *cu_seqlens,
    const void *initial_state_indices,
    const void *final_state_indices,
    bool has_bias,
    bool has_cu_seqlens,
    bool cu_seqlens_i64,
    bool initial_state_indices_i64,
    bool final_state_indices_i64,
    bool indexed_state_pool,
    size_t T_tokens,
    size_t C,
    size_t total_tokens,
    size_t request_count,
    size_t pool_size,
    ptrdiff_t out_s0,
    ptrdiff_t out_s1,
    ptrdiff_t out_s2,
    ptrdiff_t state_s0,
    ptrdiff_t state_s1,
    ptrdiff_t state_s2,
    ptrdiff_t final_s0,
    ptrdiff_t final_s1,
    ptrdiff_t final_s2,
    ptrdiff_t qkv_s0,
    ptrdiff_t qkv_s1,
    ptrdiff_t qkv_s2,
    ptrdiff_t weight_s0,
    ptrdiff_t weight_s1,
    ptrdiff_t weight_s2,
    ptrdiff_t bias_s0) {

    const size_t flat_token = blockIdx.x;
    const size_t channel = blockIdx.y * blockDim.x + threadIdx.x;
    if (flat_token >= total_tokens || channel >= C) {
        return;
    }

    int request = static_cast<int>(flat_token / T_tokens);
    int token_batch = request;
    int64_t token_begin = 0;
    int64_t token_end = static_cast<int64_t>(T_tokens);
    int64_t token_idx = static_cast<int64_t>(flat_token % T_tokens);
    if (has_cu_seqlens) {
        int left = 0;
        int right = static_cast<int>(request_count);
        while (left + 1 < right) {
            const int mid = (left + right) / 2;
            const int64_t offset = load_optional_index(cu_seqlens, cu_seqlens_i64, mid, 0);
            if (offset <= static_cast<int64_t>(flat_token)) {
                left = mid;
            } else {
                right = mid;
            }
        }
        request = left;
        token_begin = load_optional_index(cu_seqlens, cu_seqlens_i64, request, 0);
        token_end = load_optional_index(cu_seqlens, cu_seqlens_i64, request + 1, 0);
        token_batch = 0;
        token_idx = static_cast<int64_t>(flat_token);
        if (token_begin < 0 || token_idx < token_begin || token_idx >= token_end
            || token_end > static_cast<int64_t>(total_tokens)) {
            return;
        }
    }
    const int64_t local_token = token_idx - token_begin;

    int64_t read_slot = indexed_state_pool
                          ? load_optional_index(initial_state_indices, initial_state_indices_i64, request, request)
                          : static_cast<int64_t>(request);
    int64_t write_slot = indexed_state_pool && final_state_indices != nullptr
                           ? load_optional_index(final_state_indices, final_state_indices_i64, request, request)
                           : static_cast<int64_t>(request);
    if (read_slot < 0 || write_slot < 0 || read_slot >= static_cast<int64_t>(pool_size) || (final_state_indices != nullptr && write_slot >= static_cast<int64_t>(pool_size))) {
        return;
    }

    const ptrdiff_t state_base = static_cast<ptrdiff_t>(read_slot) * state_s0
                               + static_cast<ptrdiff_t>(channel) * state_s1;
    const ptrdiff_t weight_base = static_cast<ptrdiff_t>(channel) * weight_s0;
    (void)weight_s1;

    float acc = load_as_float(weight, weight_base)
                  * load_history_k4(conv_state, qkv, local_token, token_begin,
                                    token_batch, channel, state_base, state_s2,
                                    qkv_s0, qkv_s1, qkv_s2)
              + load_as_float(weight, weight_base + weight_s2)
                    * load_history_k4(conv_state, qkv, local_token + 1, token_begin,
                                      token_batch, channel, state_base, state_s2,
                                      qkv_s0, qkv_s1, qkv_s2)
              + load_as_float(weight, weight_base + 2 * weight_s2)
                    * load_history_k4(conv_state, qkv, local_token + 2, token_begin,
                                      token_batch, channel, state_base, state_s2,
                                      qkv_s0, qkv_s1, qkv_s2)
              + load_as_float(weight, weight_base + 3 * weight_s2)
                    * load_history_k4(conv_state, qkv, local_token + 3, token_begin,
                                      token_batch, channel, state_base, state_s2,
                                      qkv_s0, qkv_s1, qkv_s2);
    if (has_bias) {
        acc += load_as_float(bias, static_cast<ptrdiff_t>(channel) * bias_s0);
    }
    const ptrdiff_t out_off = static_cast<ptrdiff_t>(token_batch) * out_s0
                            + token_idx * out_s1
                            + static_cast<ptrdiff_t>(channel) * out_s2;
    out[out_off] = cast_from_float<T>(acc);
}

template <typename T>
__global__ void causal_conv1d_k4_state_kernel(
    T *conv_state,
    T *final_conv_state,
    const T *qkv,
    const void *cu_seqlens,
    const void *initial_state_indices,
    const void *final_state_indices,
    bool has_cu_seqlens,
    bool cu_seqlens_i64,
    bool initial_state_indices_i64,
    bool final_state_indices_i64,
    bool indexed_state_pool,
    size_t T_tokens,
    size_t C,
    size_t total_tokens,
    size_t pool_size,
    ptrdiff_t state_s0,
    ptrdiff_t state_s1,
    ptrdiff_t state_s2,
    ptrdiff_t final_s0,
    ptrdiff_t final_s1,
    ptrdiff_t final_s2,
    ptrdiff_t qkv_s0,
    ptrdiff_t qkv_s1,
    ptrdiff_t qkv_s2) {
    const int request = blockIdx.x;
    const size_t channel = blockIdx.y * blockDim.x + threadIdx.x;
    if (channel >= C) {
        return;
    }

    int64_t token_begin = 0;
    int64_t token_end = static_cast<int64_t>(T_tokens);
    int token_batch = request;
    if (has_cu_seqlens) {
        token_begin = load_optional_index(cu_seqlens, cu_seqlens_i64, request, 0);
        token_end = load_optional_index(cu_seqlens, cu_seqlens_i64, request + 1, 0);
        token_batch = 0;
        if (token_begin < 0 || token_end < token_begin
            || token_end > static_cast<int64_t>(total_tokens)) {
            return;
        }
    }
    const int64_t request_len = token_end - token_begin;

    const int64_t read_slot = indexed_state_pool
                                ? load_optional_index(initial_state_indices,
                                                      initial_state_indices_i64,
                                                      request, request)
                                : static_cast<int64_t>(request);
    const int64_t write_slot = indexed_state_pool && final_state_indices != nullptr
                                 ? load_optional_index(final_state_indices,
                                                       final_state_indices_i64,
                                                       request, request)
                                 : static_cast<int64_t>(request);
    if (read_slot < 0 || write_slot < 0
        || read_slot >= static_cast<int64_t>(pool_size)
        || (final_state_indices != nullptr
            && write_slot >= static_cast<int64_t>(pool_size))) {
        return;
    }

    const ptrdiff_t state_base = static_cast<ptrdiff_t>(read_slot) * state_s0
                               + static_cast<ptrdiff_t>(channel) * state_s1;

    T *final_target = final_state_indices != nullptr ? conv_state : final_conv_state;
    ptrdiff_t final_base;
    ptrdiff_t final_stride0;
    ptrdiff_t final_stride1;
    ptrdiff_t final_stride2;
    if (final_state_indices != nullptr) {
        final_base = static_cast<ptrdiff_t>(write_slot) * state_s0 + static_cast<ptrdiff_t>(channel) * state_s1;
        final_stride0 = state_s0;
        final_stride1 = state_s1;
        final_stride2 = state_s2;
    } else {
        final_base = static_cast<ptrdiff_t>(request) * final_s0 + static_cast<ptrdiff_t>(channel) * final_s1;
        final_stride0 = final_s0;
        final_stride1 = final_s1;
        final_stride2 = final_s2;
    }
    (void)final_stride0;
    (void)final_stride1;

    final_target[final_base] = cast_from_float<T>(
        load_history_k4(conv_state, qkv, request_len, token_begin, token_batch, channel, state_base, state_s2, qkv_s0, qkv_s1, qkv_s2));
    final_target[final_base + final_stride2] = cast_from_float<T>(
        load_history_k4(conv_state, qkv, request_len + 1, token_begin, token_batch, channel, state_base, state_s2, qkv_s0, qkv_s1, qkv_s2));
    final_target[final_base + 2 * final_stride2] = cast_from_float<T>(
        load_history_k4(conv_state, qkv, request_len + 2, token_begin, token_batch, channel, state_base, state_s2, qkv_s0, qkv_s1, qkv_s2));
}

template <typename T>
__global__ void causal_conv1d_k4_decode_kernel(
    T *out,
    T *conv_state,
    T *final_conv_state,
    const T *qkv,
    const T *weight,
    const T *bias,
    const void *initial_state_indices,
    const void *final_state_indices,
    bool initial_state_indices_i64,
    bool final_state_indices_i64,
    bool indexed_state_pool,
    bool has_cu_seqlens,
    bool has_bias,
    size_t C,
    size_t pool_size,
    ptrdiff_t out_s0,
    ptrdiff_t out_s1,
    ptrdiff_t out_s2,
    ptrdiff_t state_s0,
    ptrdiff_t state_s1,
    ptrdiff_t state_s2,
    ptrdiff_t final_s0,
    ptrdiff_t final_s1,
    ptrdiff_t final_s2,
    ptrdiff_t qkv_s0,
    ptrdiff_t qkv_s1,
    ptrdiff_t qkv_s2,
    ptrdiff_t weight_s0,
    ptrdiff_t weight_s2,
    ptrdiff_t bias_s0) {
    const size_t request = blockIdx.x;
    const size_t channel = blockIdx.y * blockDim.x + threadIdx.x;
    if (channel >= C) {
        return;
    }

    const int64_t read_slot = indexed_state_pool
                                ? load_optional_index(initial_state_indices,
                                                      initial_state_indices_i64,
                                                      request, request)
                                : static_cast<int64_t>(request);
    const int64_t write_slot = indexed_state_pool && final_state_indices != nullptr
                                 ? load_optional_index(final_state_indices,
                                                       final_state_indices_i64,
                                                       request, request)
                                 : static_cast<int64_t>(request);
    if (read_slot < 0 || write_slot < 0
        || read_slot >= static_cast<int64_t>(pool_size)
        || (final_state_indices != nullptr
            && write_slot >= static_cast<int64_t>(pool_size))) {
        return;
    }

    const ptrdiff_t state_base = static_cast<ptrdiff_t>(read_slot) * state_s0
                               + static_cast<ptrdiff_t>(channel) * state_s1;
    const int token_batch = has_cu_seqlens ? 0 : static_cast<int>(request);
    const int64_t token_index = has_cu_seqlens
                                  ? static_cast<int64_t>(request)
                                  : 0;
    const ptrdiff_t qkv_offset = static_cast<ptrdiff_t>(token_batch) * qkv_s0
                               + token_index * qkv_s1
                               + static_cast<ptrdiff_t>(channel) * qkv_s2;
    const float s0 = load_as_float(conv_state, state_base);
    const float s1 = load_as_float(conv_state, state_base + state_s2);
    const float s2 = load_as_float(conv_state, state_base + 2 * state_s2);
    const float token = load_as_float(qkv, qkv_offset);
    const ptrdiff_t weight_base = static_cast<ptrdiff_t>(channel) * weight_s0;
    float value = load_as_float(weight, weight_base) * s0
                + load_as_float(weight, weight_base + weight_s2) * s1
                + load_as_float(weight, weight_base + 2 * weight_s2) * s2
                + load_as_float(weight, weight_base + 3 * weight_s2) * token;
    if (has_bias) {
        value += load_as_float(bias, static_cast<ptrdiff_t>(channel) * bias_s0);
    }
    const ptrdiff_t out_offset = static_cast<ptrdiff_t>(token_batch) * out_s0
                               + token_index * out_s1
                               + static_cast<ptrdiff_t>(channel) * out_s2;
    out[out_offset] = cast_from_float<T>(value);

    T *final_target = final_state_indices != nullptr ? conv_state : final_conv_state;
    const ptrdiff_t final_base = final_state_indices != nullptr
                                   ? static_cast<ptrdiff_t>(write_slot) * state_s0
                                         + static_cast<ptrdiff_t>(channel) * state_s1
                                   : static_cast<ptrdiff_t>(request) * final_s0
                                         + static_cast<ptrdiff_t>(channel) * final_s1;
    const ptrdiff_t final_stride = final_state_indices != nullptr ? state_s2 : final_s2;
    final_target[final_base] = cast_from_float<T>(s1);
    final_target[final_base + final_stride] = cast_from_float<T>(s2);
    final_target[final_base + 2 * final_stride] = cast_from_float<T>(token);
}

template <typename T>
infiniStatus_t launch_k4(
    const CausalConv1dInfo &info,
    void *out,
    void *conv_state,
    void *final_conv_state,
    const void *qkv,
    const void *weight,
    const void *bias,
    const void *cu_seqlens,
    const void *initial_state_indices,
    const void *final_state_indices,
    cudaStream_t stream) {

    constexpr unsigned int threads = 256;
    const bool decode = info.total_tokens == info.request_count;
    if (decode) {
        const dim3 decode_grid(
            static_cast<unsigned int>(info.request_count),
            static_cast<unsigned int>((info.C + threads - 1) / threads));
        causal_conv1d_k4_decode_kernel<T><<<decode_grid, threads, 0, stream>>>(
            static_cast<T *>(out),
            static_cast<T *>(conv_state),
            static_cast<T *>(final_conv_state),
            static_cast<const T *>(qkv),
            static_cast<const T *>(weight),
            static_cast<const T *>(bias),
            initial_state_indices,
            final_state_indices,
            info.initial_state_indices_dtype == INFINI_DTYPE_I64,
            info.final_state_indices_dtype == INFINI_DTYPE_I64,
            info.indexed_state_pool,
            info.has_cu_seqlens,
            info.has_bias,
            info.C,
            info.pool_size,
            info.out_strides[0],
            info.out_strides[1],
            info.out_strides[2],
            info.conv_state_strides[0],
            info.conv_state_strides[1],
            info.conv_state_strides[2],
            info.final_conv_state_strides.empty() ? 0 : info.final_conv_state_strides[0],
            info.final_conv_state_strides.empty() ? 0 : info.final_conv_state_strides[1],
            info.final_conv_state_strides.empty() ? 0 : info.final_conv_state_strides[2],
            info.qkv_strides[0],
            info.qkv_strides[1],
            info.qkv_strides[2],
            info.weight_strides[0],
            info.weight_strides[2],
            info.bias_strides.empty() ? 0 : info.bias_strides[0]);
        CHECK_CUDA(cudaGetLastError());
        return INFINI_STATUS_SUCCESS;
    }

    const dim3 output_grid(
        static_cast<unsigned int>(info.total_tokens),
        static_cast<unsigned int>((info.C + threads - 1) / threads));
    causal_conv1d_k4_prefill_kernel<T><<<output_grid, threads, 0, stream>>>(
        static_cast<T *>(out),
        static_cast<T *>(conv_state),
        static_cast<T *>(final_conv_state),
        static_cast<const T *>(qkv),
        static_cast<const T *>(weight),
        static_cast<const T *>(bias),
        cu_seqlens,
        initial_state_indices,
        final_state_indices,
        info.has_bias,
        info.has_cu_seqlens,
        info.cu_seqlens_dtype == INFINI_DTYPE_I64,
        info.initial_state_indices_dtype == INFINI_DTYPE_I64,
        info.final_state_indices_dtype == INFINI_DTYPE_I64,
        info.indexed_state_pool,
        info.T,
        info.C,
        info.total_tokens,
        info.request_count,
        info.pool_size,
        info.out_strides[0],
        info.out_strides[1],
        info.out_strides[2],
        info.conv_state_strides[0],
        info.conv_state_strides[1],
        info.conv_state_strides[2],
        info.final_conv_state_strides.empty() ? 0 : info.final_conv_state_strides[0],
        info.final_conv_state_strides.empty() ? 0 : info.final_conv_state_strides[1],
        info.final_conv_state_strides.empty() ? 0 : info.final_conv_state_strides[2],
        info.qkv_strides[0],
        info.qkv_strides[1],
        info.qkv_strides[2],
        info.weight_strides[0],
        info.weight_strides[1],
        info.weight_strides[2],
        info.bias_strides.empty() ? 0 : info.bias_strides[0]);
    CHECK_CUDA(cudaGetLastError());

    const dim3 state_grid(
        static_cast<unsigned int>(info.request_count),
        static_cast<unsigned int>((info.C + threads - 1) / threads));
    causal_conv1d_k4_state_kernel<T><<<state_grid, threads, 0, stream>>>(
        static_cast<T *>(conv_state),
        static_cast<T *>(final_conv_state),
        static_cast<const T *>(qkv),
        cu_seqlens,
        initial_state_indices,
        final_state_indices,
        info.has_cu_seqlens,
        info.cu_seqlens_dtype == INFINI_DTYPE_I64,
        info.initial_state_indices_dtype == INFINI_DTYPE_I64,
        info.final_state_indices_dtype == INFINI_DTYPE_I64,
        info.indexed_state_pool,
        info.T,
        info.C,
        info.total_tokens,
        info.pool_size,
        info.conv_state_strides[0],
        info.conv_state_strides[1],
        info.conv_state_strides[2],
        info.final_conv_state_strides.empty() ? 0 : info.final_conv_state_strides[0],
        info.final_conv_state_strides.empty() ? 0 : info.final_conv_state_strides[1],
        info.final_conv_state_strides.empty() ? 0 : info.final_conv_state_strides[2],
        info.qkv_strides[0],
        info.qkv_strides[1],
        info.qkv_strides[2]);
    CHECK_CUDA(cudaGetLastError());
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t launch_by_kernel_size(
    const CausalConv1dInfo &info,
    void *out,
    void *conv_state,
    void *final_conv_state,
    const void *qkv,
    const void *weight,
    const void *bias,
    const void *cu_seqlens,
    const void *initial_state_indices,
    const void *final_state_indices,
    cudaStream_t stream) {

    if (info.kernel_size != 4) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    switch (info.data_dtype) {
    case INFINI_DTYPE_F32:
        return launch_k4<float>(info, out, conv_state, final_conv_state, qkv, weight, bias, cu_seqlens, initial_state_indices, final_state_indices, stream);
    case INFINI_DTYPE_F16:
        return launch_k4<half>(info, out, conv_state, final_conv_state, qkv, weight, bias, cu_seqlens, initial_state_indices, final_state_indices, stream);
    case INFINI_DTYPE_BF16:
        return launch_k4<__nv_bfloat16>(info, out, conv_state, final_conv_state, qkv, weight, bias, cu_seqlens, initial_state_indices, final_state_indices, stream);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *out,
    void *conv_state,
    void *final_conv_state,
    const void *qkv,
    const void *weight,
    const void *bias,
    const void *cu_seqlens,
    const void *initial_state_indices,
    const void *final_state_indices,
    void *stream) const {

    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    return launch_by_kernel_size(_info, out, conv_state, final_conv_state, qkv, weight, bias, cu_seqlens, initial_state_indices, final_state_indices, cuda_stream);
}

} // namespace nvidia
} // namespace causal_conv1d
} // namespace op
