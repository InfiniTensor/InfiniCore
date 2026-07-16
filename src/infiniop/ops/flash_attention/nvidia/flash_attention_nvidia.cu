#include "flash_attention_nvidia.cuh"

#include <cmath>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

using cuda_bfloat16 = nv_bfloat16;

namespace {

constexpr int BLOCK_SIZE = 256;

template <typename T>
__device__ __forceinline__ float load_as_float(const T *ptr, ptrdiff_t offset) {
    return static_cast<float>(ptr[offset]);
}

template <>
__device__ __forceinline__ float load_as_float<half>(const half *ptr, ptrdiff_t offset) {
    return __half2float(ptr[offset]);
}

template <>
__device__ __forceinline__ float load_as_float<cuda_bfloat16>(const cuda_bfloat16 *ptr, ptrdiff_t offset) {
    return __bfloat162float(ptr[offset]);
}

template <typename T>
__device__ __forceinline__ void store_from_float(T *ptr, ptrdiff_t offset, float value) {
    ptr[offset] = static_cast<T>(value);
}

template <>
__device__ __forceinline__ void store_from_float<half>(half *ptr, ptrdiff_t offset, float value) {
    ptr[offset] = __float2half_rn(value);
}

template <>
__device__ __forceinline__ void store_from_float<cuda_bfloat16>(cuda_bfloat16 *ptr, ptrdiff_t offset, float value) {
    ptr[offset] = __float2bfloat16_rn(value);
}

template <typename LenT>
__device__ __forceinline__ size_t load_total_kv_len(const void *total_kv_len,
                                                    size_t batch,
                                                    ptrdiff_t stride) {
    auto ptr = static_cast<const LenT *>(total_kv_len);
    auto value = ptr[batch * stride];
    return value > 0 ? static_cast<size_t>(value) : 0;
}

template <typename LenT>
__device__ __forceinline__ size_t total_len_dispatch(const void *total_kv_len,
                                                     size_t batch,
                                                     ptrdiff_t stride) {
    return load_total_kv_len<LenT>(total_kv_len, batch, stride);
}

template <typename T, typename LenT>
__global__ void flashAttentionKernel(
    T *out,
    const T *q,
    const T *k,
    const T *v,
    const void *total_kv_len,
    float scale,
    bool is_causal,
    size_t batch_size,
    size_t num_q_heads,
    size_t query_len,
    size_t head_dim,
    size_t num_kv_heads,
    size_t max_kv_len,
    ptrdiff_t out_stride_b,
    ptrdiff_t out_stride_h,
    ptrdiff_t out_stride_s,
    ptrdiff_t out_stride_d,
    ptrdiff_t q_stride_b,
    ptrdiff_t q_stride_h,
    ptrdiff_t q_stride_s,
    ptrdiff_t q_stride_d,
    ptrdiff_t k_stride_b,
    ptrdiff_t k_stride_h,
    ptrdiff_t k_stride_s,
    ptrdiff_t k_stride_d,
    ptrdiff_t v_stride_b,
    ptrdiff_t v_stride_h,
    ptrdiff_t v_stride_s,
    ptrdiff_t v_stride_d,
    ptrdiff_t total_kv_stride) {

    const size_t row = blockIdx.x;
    const size_t query_idx = row % query_len;
    const size_t q_head = (row / query_len) % num_q_heads;
    const size_t batch = row / (query_len * num_q_heads);
    if (batch >= batch_size) {
        return;
    }

    const size_t group_size = num_q_heads / num_kv_heads;
    const size_t kv_head = q_head / group_size;
    size_t total_len = total_len_dispatch<LenT>(total_kv_len, batch, total_kv_stride);
    total_len = total_len < max_kv_len ? total_len : max_kv_len;

    size_t visible_len = total_len;
    if (is_causal) {
        const size_t first_query_key = total_len > query_len ? total_len - query_len : 0;
        const size_t causal_len = first_query_key + query_idx + 1;
        visible_len = visible_len < causal_len ? visible_len : causal_len;
    }

    const size_t value_dim = threadIdx.x;
    __shared__ float shared[BLOCK_SIZE];
    __shared__ float row_max_shared;
    __shared__ float row_sum_shared;

    if (visible_len == 0) {
        if (value_dim < head_dim) {
            const auto out_offset = static_cast<ptrdiff_t>(batch) * out_stride_b
                                  + static_cast<ptrdiff_t>(q_head) * out_stride_h
                                  + static_cast<ptrdiff_t>(query_idx) * out_stride_s
                                  + static_cast<ptrdiff_t>(value_dim) * out_stride_d;
            store_from_float(out, out_offset, 0.0f);
        }
        return;
    }

    float row_max = -INFINITY;
    for (size_t key_idx = 0; key_idx < visible_len; ++key_idx) {
        float partial = 0.0f;
        for (size_t dim = threadIdx.x; dim < head_dim; dim += BLOCK_SIZE) {
            const auto q_offset = static_cast<ptrdiff_t>(batch) * q_stride_b
                                + static_cast<ptrdiff_t>(q_head) * q_stride_h
                                + static_cast<ptrdiff_t>(query_idx) * q_stride_s
                                + static_cast<ptrdiff_t>(dim) * q_stride_d;
            const auto k_offset = static_cast<ptrdiff_t>(batch) * k_stride_b
                                + static_cast<ptrdiff_t>(kv_head) * k_stride_h
                                + static_cast<ptrdiff_t>(key_idx) * k_stride_s
                                + static_cast<ptrdiff_t>(dim) * k_stride_d;
            partial += load_as_float(q, q_offset) * load_as_float(k, k_offset);
        }
        shared[threadIdx.x] = partial;
        __syncthreads();
        for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                shared[threadIdx.x] += shared[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            row_max = fmaxf(row_max, shared[0] * scale);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        row_max_shared = row_max;
    }
    __syncthreads();

    float acc = 0.0f;
    float row_sum = 0.0f;
    for (size_t key_idx = 0; key_idx < visible_len; ++key_idx) {
        float partial = 0.0f;
        for (size_t dim = threadIdx.x; dim < head_dim; dim += BLOCK_SIZE) {
            const auto q_offset = static_cast<ptrdiff_t>(batch) * q_stride_b
                                + static_cast<ptrdiff_t>(q_head) * q_stride_h
                                + static_cast<ptrdiff_t>(query_idx) * q_stride_s
                                + static_cast<ptrdiff_t>(dim) * q_stride_d;
            const auto k_offset = static_cast<ptrdiff_t>(batch) * k_stride_b
                                + static_cast<ptrdiff_t>(kv_head) * k_stride_h
                                + static_cast<ptrdiff_t>(key_idx) * k_stride_s
                                + static_cast<ptrdiff_t>(dim) * k_stride_d;
            partial += load_as_float(q, q_offset) * load_as_float(k, k_offset);
        }
        shared[threadIdx.x] = partial;
        __syncthreads();
        for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                shared[threadIdx.x] += shared[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            row_sum += expf(shared[0] * scale - row_max_shared);
        }
        const float numerator = expf(shared[0] * scale - row_max_shared);
        if (value_dim < head_dim) {
            const auto v_offset = static_cast<ptrdiff_t>(batch) * v_stride_b
                                + static_cast<ptrdiff_t>(kv_head) * v_stride_h
                                + static_cast<ptrdiff_t>(key_idx) * v_stride_s
                                + static_cast<ptrdiff_t>(value_dim) * v_stride_d;
            acc += numerator * load_as_float(v, v_offset);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        row_sum_shared = row_sum;
    }
    __syncthreads();

    if (value_dim < head_dim) {
        const auto out_offset = static_cast<ptrdiff_t>(batch) * out_stride_b
                              + static_cast<ptrdiff_t>(q_head) * out_stride_h
                              + static_cast<ptrdiff_t>(query_idx) * out_stride_s
                              + static_cast<ptrdiff_t>(value_dim) * out_stride_d;
        store_from_float(out, out_offset, acc / row_sum_shared);
    }
}

template <typename T, typename LenT>
__global__ void flashAttentionDecodeSplitKernel(
    float *partial,
    const T *q,
    const T *k,
    const T *v,
    const void *total_kv_len,
    float scale,
    size_t batch_size,
    size_t num_q_heads,
    size_t head_dim,
    size_t num_kv_heads,
    size_t max_kv_len,
    size_t split_size,
    size_t num_splits,
    ptrdiff_t q_stride_b,
    ptrdiff_t q_stride_h,
    ptrdiff_t q_stride_s,
    ptrdiff_t q_stride_d,
    ptrdiff_t k_stride_b,
    ptrdiff_t k_stride_h,
    ptrdiff_t k_stride_s,
    ptrdiff_t k_stride_d,
    ptrdiff_t v_stride_b,
    ptrdiff_t v_stride_h,
    ptrdiff_t v_stride_s,
    ptrdiff_t v_stride_d,
    ptrdiff_t total_kv_stride) {

    const size_t row = blockIdx.x;
    const size_t split = blockIdx.y;
    const size_t q_head = row % num_q_heads;
    const size_t batch = row / num_q_heads;
    if (batch >= batch_size) {
        return;
    }

    const size_t group_size = num_q_heads / num_kv_heads;
    const size_t kv_head = q_head / group_size;
    size_t total_len = total_len_dispatch<LenT>(total_kv_len, batch, total_kv_stride);
    total_len = total_len < max_kv_len ? total_len : max_kv_len;

    const size_t begin = split * split_size;
    const size_t end = min(begin + split_size, total_len);
    const size_t value_dim = threadIdx.x;
    const size_t partial_stride = head_dim + 2;
    float *partial_base = partial + (split * batch_size * num_q_heads + row) * partial_stride;

    __shared__ float shared[BLOCK_SIZE];
    __shared__ float row_max_shared;
    __shared__ float row_sum_shared;

    if (begin >= end) {
        if (threadIdx.x == 0) {
            partial_base[0] = -INFINITY;
            partial_base[1] = 0.0f;
        }
        for (size_t dim = value_dim; dim < head_dim; dim += BLOCK_SIZE) {
            partial_base[2 + dim] = 0.0f;
        }
        return;
    }

    float row_max = -INFINITY;
    for (size_t key_idx = begin; key_idx < end; ++key_idx) {
        float partial_dot = 0.0f;
        for (size_t dim = threadIdx.x; dim < head_dim; dim += BLOCK_SIZE) {
            const auto q_offset = static_cast<ptrdiff_t>(batch) * q_stride_b
                                + static_cast<ptrdiff_t>(q_head) * q_stride_h
                                + static_cast<ptrdiff_t>(dim) * q_stride_d;
            const auto k_offset = static_cast<ptrdiff_t>(batch) * k_stride_b
                                + static_cast<ptrdiff_t>(kv_head) * k_stride_h
                                + static_cast<ptrdiff_t>(key_idx) * k_stride_s
                                + static_cast<ptrdiff_t>(dim) * k_stride_d;
            partial_dot += load_as_float(q, q_offset) * load_as_float(k, k_offset);
        }
        shared[threadIdx.x] = partial_dot;
        __syncthreads();
        for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                shared[threadIdx.x] += shared[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            row_max = fmaxf(row_max, shared[0] * scale);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        row_max_shared = row_max;
        partial_base[0] = row_max;
    }
    __syncthreads();

    float acc = 0.0f;
    float row_sum = 0.0f;
    for (size_t key_idx = begin; key_idx < end; ++key_idx) {
        float partial_dot = 0.0f;
        for (size_t dim = threadIdx.x; dim < head_dim; dim += BLOCK_SIZE) {
            const auto q_offset = static_cast<ptrdiff_t>(batch) * q_stride_b
                                + static_cast<ptrdiff_t>(q_head) * q_stride_h
                                + static_cast<ptrdiff_t>(dim) * q_stride_d;
            const auto k_offset = static_cast<ptrdiff_t>(batch) * k_stride_b
                                + static_cast<ptrdiff_t>(kv_head) * k_stride_h
                                + static_cast<ptrdiff_t>(key_idx) * k_stride_s
                                + static_cast<ptrdiff_t>(dim) * k_stride_d;
            partial_dot += load_as_float(q, q_offset) * load_as_float(k, k_offset);
        }
        shared[threadIdx.x] = partial_dot;
        __syncthreads();
        for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                shared[threadIdx.x] += shared[threadIdx.x + stride];
            }
            __syncthreads();
        }
        const float numerator = expf(shared[0] * scale - row_max_shared);
        if (threadIdx.x == 0) {
            row_sum += numerator;
        }
        if (value_dim < head_dim) {
            const auto v_offset = static_cast<ptrdiff_t>(batch) * v_stride_b
                                + static_cast<ptrdiff_t>(kv_head) * v_stride_h
                                + static_cast<ptrdiff_t>(key_idx) * v_stride_s
                                + static_cast<ptrdiff_t>(value_dim) * v_stride_d;
            acc += numerator * load_as_float(v, v_offset);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        row_sum_shared = row_sum;
        partial_base[1] = row_sum;
    }
    __syncthreads();

    for (size_t dim = value_dim; dim < head_dim; dim += BLOCK_SIZE) {
        partial_base[2 + dim] = (dim == value_dim) ? acc : 0.0f;
    }
}

template <typename T>
__global__ void flashAttentionDecodeCombineKernel(
    T *out,
    const float *partial,
    size_t batch_size,
    size_t num_q_heads,
    size_t head_dim,
    size_t num_splits,
    ptrdiff_t out_stride_b,
    ptrdiff_t out_stride_h,
    ptrdiff_t out_stride_d) {

    const size_t row = blockIdx.x;
    const size_t q_head = row % num_q_heads;
    const size_t batch = row / num_q_heads;
    if (batch >= batch_size) {
        return;
    }

    const size_t partial_stride = head_dim + 2;
    __shared__ float global_max;
    __shared__ float global_sum;

    if (threadIdx.x == 0) {
        float m = -INFINITY;
        for (size_t split = 0; split < num_splits; ++split) {
            const float *partial_base = partial + (split * batch_size * num_q_heads + row) * partial_stride;
            m = fmaxf(m, partial_base[0]);
        }
        global_max = m;

        float s = 0.0f;
        for (size_t split = 0; split < num_splits; ++split) {
            const float *partial_base = partial + (split * batch_size * num_q_heads + row) * partial_stride;
            if (partial_base[0] != -INFINITY) {
                s += expf(partial_base[0] - m) * partial_base[1];
            }
        }
        global_sum = s;
    }
    __syncthreads();

    for (size_t dim = threadIdx.x; dim < head_dim; dim += BLOCK_SIZE) {
        float acc = 0.0f;
        for (size_t split = 0; split < num_splits; ++split) {
            const float *partial_base = partial + (split * batch_size * num_q_heads + row) * partial_stride;
            if (partial_base[0] != -INFINITY) {
                acc += expf(partial_base[0] - global_max) * partial_base[2 + dim];
            }
        }
        const auto out_offset = static_cast<ptrdiff_t>(batch) * out_stride_b
                              + static_cast<ptrdiff_t>(q_head) * out_stride_h
                              + static_cast<ptrdiff_t>(dim) * out_stride_d;
        store_from_float(out, out_offset, global_sum > 0.0f ? acc / global_sum : 0.0f);
    }
}

template <typename T, typename LenT>
infiniStatus_t launchFlashAttentionDecodeSplit(
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *q,
    const void *k,
    const void *v,
    const void *total_kv_len,
    float scale,
    size_t batch_size,
    size_t num_q_heads,
    size_t head_dim,
    size_t num_kv_heads,
    size_t max_kv_len,
    const std::vector<ptrdiff_t> &out_strides,
    const std::vector<ptrdiff_t> &q_strides,
    const std::vector<ptrdiff_t> &k_strides,
    const std::vector<ptrdiff_t> &v_strides,
    const std::vector<ptrdiff_t> &total_kv_strides,
    void *stream) {
    constexpr size_t SPLIT_SIZE = 64;
    const size_t num_splits = (max_kv_len + SPLIT_SIZE - 1) / SPLIT_SIZE;
    const size_t rows = batch_size * num_q_heads;
    const size_t required = rows * num_splits * (head_dim + 2) * sizeof(float);
    if (workspace_size < required) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    auto partial = static_cast<float *>(workspace);
    flashAttentionDecodeSplitKernel<T, LenT><<<dim3(static_cast<unsigned int>(rows), static_cast<unsigned int>(num_splits)), BLOCK_SIZE, 0, cuda_stream>>>(
        partial,
        static_cast<const T *>(q),
        static_cast<const T *>(k),
        static_cast<const T *>(v),
        total_kv_len,
        scale,
        batch_size,
        num_q_heads,
        head_dim,
        num_kv_heads,
        max_kv_len,
        SPLIT_SIZE,
        num_splits,
        q_strides[0],
        q_strides[1],
        q_strides[2],
        q_strides[3],
        k_strides[0],
        k_strides[1],
        k_strides[2],
        k_strides[3],
        v_strides[0],
        v_strides[1],
        v_strides[2],
        v_strides[3],
        total_kv_strides.empty() ? 1 : total_kv_strides[0]);
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        return INFINI_STATUS_INTERNAL_ERROR;
    }
    flashAttentionDecodeCombineKernel<T><<<static_cast<unsigned int>(rows), BLOCK_SIZE, 0, cuda_stream>>>(
        static_cast<T *>(out),
        partial,
        batch_size,
        num_q_heads,
        head_dim,
        num_splits,
        out_strides[0],
        out_strides[1],
        out_strides[3]);
    err = cudaGetLastError();
    return err == cudaSuccess ? INFINI_STATUS_SUCCESS : INFINI_STATUS_INTERNAL_ERROR;
}

template <typename T, typename LenT>
infiniStatus_t launchFlashAttention(
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *q,
    const void *k,
    const void *v,
    const void *total_kv_len,
    float scale,
    bool is_causal,
    size_t batch_size,
    size_t num_q_heads,
    size_t query_len,
    size_t head_dim,
    size_t num_kv_heads,
    size_t max_kv_len,
    const std::vector<ptrdiff_t> &out_strides,
    const std::vector<ptrdiff_t> &q_strides,
    const std::vector<ptrdiff_t> &k_strides,
    const std::vector<ptrdiff_t> &v_strides,
    const std::vector<ptrdiff_t> &total_kv_strides,
    void *stream) {
    const size_t rows = batch_size * num_q_heads * query_len;
    if (rows == 0 || head_dim == 0) {
        return INFINI_STATUS_SUCCESS;
    }
    if (query_len == 1 && max_kv_len > 128) {
        return launchFlashAttentionDecodeSplit<T, LenT>(workspace, workspace_size, out, q, k, v, total_kv_len, scale,
                                                        batch_size, num_q_heads, head_dim, num_kv_heads, max_kv_len,
                                                        out_strides, q_strides, k_strides, v_strides, total_kv_strides, stream);
    }
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    flashAttentionKernel<T, LenT><<<static_cast<unsigned int>(rows), BLOCK_SIZE, 0, cuda_stream>>>(
        static_cast<T *>(out),
        static_cast<const T *>(q),
        static_cast<const T *>(k),
        static_cast<const T *>(v),
        total_kv_len,
        scale,
        is_causal,
        batch_size,
        num_q_heads,
        query_len,
        head_dim,
        num_kv_heads,
        max_kv_len,
        out_strides[0],
        out_strides[1],
        out_strides[2],
        out_strides[3],
        q_strides[0],
        q_strides[1],
        q_strides[2],
        q_strides[3],
        k_strides[0],
        k_strides[1],
        k_strides[2],
        k_strides[3],
        v_strides[0],
        v_strides[1],
        v_strides[2],
        v_strides[3],
        total_kv_strides.empty() ? 1 : total_kv_strides[0]);
    auto err = cudaGetLastError();
    return err == cudaSuccess ? INFINI_STATUS_SUCCESS : INFINI_STATUS_INTERNAL_ERROR;
}

template <typename T>
infiniStatus_t dispatchLenType(
    infiniDtype_t len_dtype,
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *q,
    const void *k,
    const void *v,
    const void *total_kv_len,
    float scale,
    bool is_causal,
    size_t batch_size,
    size_t num_q_heads,
    size_t query_len,
    size_t head_dim,
    size_t num_kv_heads,
    size_t max_kv_len,
    const std::vector<ptrdiff_t> &out_strides,
    const std::vector<ptrdiff_t> &q_strides,
    const std::vector<ptrdiff_t> &k_strides,
    const std::vector<ptrdiff_t> &v_strides,
    const std::vector<ptrdiff_t> &total_kv_strides,
    void *stream) {
    switch (len_dtype) {
    case INFINI_DTYPE_I32:
        return launchFlashAttention<T, int32_t>(workspace, workspace_size, out, q, k, v, total_kv_len, scale, is_causal,
                                                batch_size, num_q_heads, query_len, head_dim, num_kv_heads, max_kv_len,
                                                out_strides, q_strides, k_strides, v_strides, total_kv_strides, stream);
    case INFINI_DTYPE_I64:
        return launchFlashAttention<T, int64_t>(workspace, workspace_size, out, q, k, v, total_kv_len, scale, is_causal,
                                                batch_size, num_q_heads, query_len, head_dim, num_kv_heads, max_kv_len,
                                                out_strides, q_strides, k_strides, v_strides, total_kv_strides, stream);
    case INFINI_DTYPE_U32:
        return launchFlashAttention<T, uint32_t>(workspace, workspace_size, out, q, k, v, total_kv_len, scale, is_causal,
                                                 batch_size, num_q_heads, query_len, head_dim, num_kv_heads, max_kv_len,
                                                 out_strides, q_strides, k_strides, v_strides, total_kv_strides, stream);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace

namespace op::flash_attention::nvidia {

Descriptor::Descriptor(infiniopHandle_t handle,
                       infiniopTensorDescriptor_t out_desc,
                       infiniopTensorDescriptor_t q_desc,
                       infiniopTensorDescriptor_t k_desc,
                       infiniopTensorDescriptor_t v_desc,
                       infiniopTensorDescriptor_t total_kv_len,
                       float scale,
                       char is_causal)
    : InfiniopDescriptor{handle->device, handle->device_id},
      _out_shape{out_desc->shape()},
      _out_strides{out_desc->strides()},
      _query_shape{q_desc->shape()},
      _query_strides{q_desc->strides()},
      _key_shape{k_desc->shape()},
      _key_strides{k_desc->strides()},
      _value_shape{v_desc->shape()},
      _value_strides{v_desc->strides()},
      _total_kv_shape{total_kv_len->shape()},
      _total_kv_strides{total_kv_len->strides()},
      _dtype{q_desc->dtype()},
      _total_kv_dtype{total_kv_len->dtype()},
      _scale{scale},
      _is_causal{is_causal} {
}

size_t Descriptor::get_workspace_size() const {
    const auto query_len = _query_shape[2];
    const auto head_dim = _query_shape[3];
    const auto max_kv_len = _key_shape[2];
    if (query_len != 1 || max_kv_len <= 128) {
        return 0;
    }
    constexpr size_t SPLIT_SIZE = 64;
    const auto num_splits = (max_kv_len + SPLIT_SIZE - 1) / SPLIT_SIZE;
    const auto rows = _query_shape[0] * _query_shape[1];
    return rows * num_splits * (head_dim + 2) * sizeof(float);
}

infiniStatus_t Descriptor::create(infiniopHandle_t handle,
                                  Descriptor **desc,
                                  infiniopTensorDescriptor_t out_desc,
                                  infiniopTensorDescriptor_t q_desc,
                                  infiniopTensorDescriptor_t k_desc,
                                  infiniopTensorDescriptor_t v_desc,
                                  infiniopTensorDescriptor_t total_kv_len,
                                  float scale,
                                  char is_causal) {
    if (q_desc->ndim() != 4 || k_desc->ndim() != 4 || v_desc->ndim() != 4 || out_desc->ndim() != 4) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }
    if (q_desc->dtype() != k_desc->dtype() || q_desc->dtype() != v_desc->dtype() || q_desc->dtype() != out_desc->dtype()) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    if (q_desc->dim(0) != k_desc->dim(0) || q_desc->dim(0) != v_desc->dim(0)
        || q_desc->dim(0) != out_desc->dim(0)
        || q_desc->dim(2) != out_desc->dim(2)
        || q_desc->dim(3) != k_desc->dim(3)
        || q_desc->dim(3) != v_desc->dim(3)
        || q_desc->dim(3) != out_desc->dim(3)
        || k_desc->dim(1) != v_desc->dim(1)
        || k_desc->dim(2) != v_desc->dim(2)) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }
    if (q_desc->dim(1) % k_desc->dim(1) != 0) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }
    if (total_kv_len->ndim() != 1 || total_kv_len->dim(0) < q_desc->dim(0)) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }
    if (q_desc->dim(3) > BLOCK_SIZE) {
        return INFINI_STATUS_NOT_IMPLEMENTED;
    }

    *desc = new Descriptor{handle, out_desc, q_desc, k_desc, v_desc, total_kv_len, scale, is_causal};
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(void *workspace,
                                     size_t workspace_size,
                                     void *out,
                                     const void *q,
                                     const void *k,
                                     const void *v,
                                     const void *total_kv_len,
                                     void *stream) const {
    (void)workspace;
    (void)workspace_size;

    const auto batch_size = _query_shape[0];
    const auto num_q_heads = _query_shape[1];
    const auto query_len = _query_shape[2];
    const auto head_dim = _query_shape[3];
    const auto num_kv_heads = _key_shape[1];
    const auto max_kv_len = _key_shape[2];

    switch (_dtype) {
    case INFINI_DTYPE_F16:
        return dispatchLenType<half>(_total_kv_dtype, workspace, workspace_size, out, q, k, v, total_kv_len, _scale, _is_causal,
                                     batch_size, num_q_heads, query_len, head_dim, num_kv_heads, max_kv_len,
                                     _out_strides, _query_strides, _key_strides, _value_strides, _total_kv_strides, stream);
    case INFINI_DTYPE_BF16:
        return dispatchLenType<cuda_bfloat16>(_total_kv_dtype, workspace, workspace_size, out, q, k, v, total_kv_len, _scale, _is_causal,
                                              batch_size, num_q_heads, query_len, head_dim, num_kv_heads, max_kv_len,
                                              _out_strides, _query_strides, _key_strides, _value_strides, _total_kv_strides, stream);
    case INFINI_DTYPE_F32:
        return dispatchLenType<float>(_total_kv_dtype, workspace, workspace_size, out, q, k, v, total_kv_len, _scale, _is_causal,
                                      batch_size, num_q_heads, query_len, head_dim, num_kv_heads, max_kv_len,
                                      _out_strides, _query_strides, _key_strides, _value_strides, _total_kv_strides, stream);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::flash_attention::nvidia
