#include "../../../devices/nvidia/nvidia_common.cuh"
#include "flash_attention_nvidia.cuh"

#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../cuda/kernel.cuh"

namespace op::flash_attention::nvidia {

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
    infiniopTensorDescriptor_t q_desc,
    infiniopTensorDescriptor_t k_desc,
    infiniopTensorDescriptor_t v_desc,
    infiniopTensorDescriptor_t total_kv_len_desc,
    float scale,
    char is_causal) {

    auto result = FlashAttentionInfo::create(
        out_desc, q_desc, k_desc, v_desc, total_kv_len_desc, scale, is_causal);
    CHECK_RESULT(result);
    auto info = result.take();

    // Calculate workspace size
    size_t workspace_size = 0;

    // For Flash Attention v2, we need workspace for:
    // 1. Softmax statistics (if needed)
    // 2. Temporary buffers for block-wise computation
    size_t num_blocks_q = (info.seq_len_q + 63) / 64; // ceil division
    size_t num_blocks_kv = (info.seq_len_kv + 63) / 64;

    // Softmax statistics: each block needs lse (log sum exp) for backward
    // For forward pass only, we might need less
    workspace_size = info.batch_size * info.num_heads * num_blocks_q * sizeof(float) * 2;

    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        std::move(info),
        workspace_size,
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

// Forward declaration of kernel launcher
template <typename Tdata, typename Tcompute>
infiniStatus_t launchFlashAttentionKernel(
    const FlashAttentionInfo &info,
    void *workspace, size_t workspace_size,
    void *out,
    const void *q,
    const void *k,
    const void *v,
    const void *total_kv_len,
    cudaStream_t stream,
    int max_threads_per_block);

// Specializations for different data types
template <>
infiniStatus_t launchFlashAttentionKernel<half, float>(
    const FlashAttentionInfo &info,
    void *workspace, size_t workspace_size,
    void *out,
    const void *q,
    const void *k,
    const void *v,
    const void *total_kv_len,
    cudaStream_t stream,
    int max_threads_per_block) {

    return flash_attention_cuda::flash_attention_forward<half, float>(
        reinterpret_cast<half *>(out),
        reinterpret_cast<const half *>(q),
        reinterpret_cast<const half *>(k),
        reinterpret_cast<const half *>(v),
        reinterpret_cast<const int64_t *>(total_kv_len),
        info.batch_size,
        info.num_heads,
        info.num_kv_heads,
        info.seq_len_q,
        info.seq_len_kv,
        info.head_dim,
        info.scale,
        info.is_causal,
        info.has_variable_kv_len,
        info.q_stride_batch, info.q_stride_head, info.q_stride_seq,
        info.k_stride_batch, info.k_stride_head, info.k_stride_seq,
        info.v_stride_batch, info.v_stride_head, info.v_stride_seq,
        info.out_stride_batch, info.out_stride_head, info.out_stride_seq,
        reinterpret_cast<float *>(workspace),
        workspace_size / sizeof(float),
        stream);
}

template <>
infiniStatus_t launchFlashAttentionKernel<__nv_bfloat16, float>(
    const FlashAttentionInfo &info,
    void *workspace, size_t workspace_size,
    void *out,
    const void *q,
    const void *k,
    const void *v,
    const void *total_kv_len,
    cudaStream_t stream,
    int max_threads_per_block) {

    return flash_attention_cuda::flash_attention_forward<__nv_bfloat16, float>(
        reinterpret_cast<__nv_bfloat16 *>(out),
        reinterpret_cast<const __nv_bfloat16 *>(q),
        reinterpret_cast<const __nv_bfloat16 *>(k),
        reinterpret_cast<const __nv_bfloat16 *>(v),
        reinterpret_cast<const int64_t *>(total_kv_len),
        info.batch_size,
        info.num_heads,
        info.num_kv_heads,
        info.seq_len_q,
        info.seq_len_kv,
        info.head_dim,
        info.scale,
        info.is_causal,
        info.has_variable_kv_len,
        info.q_stride_batch, info.q_stride_head, info.q_stride_seq,
        info.k_stride_batch, info.k_stride_head, info.k_stride_seq,
        info.v_stride_batch, info.v_stride_head, info.v_stride_seq,
        info.out_stride_batch, info.out_stride_head, info.out_stride_seq,
        reinterpret_cast<float *>(workspace),
        workspace_size / sizeof(float),
        stream);
}

template <>
infiniStatus_t launchFlashAttentionKernel<float, float>(
    const FlashAttentionInfo &info,
    void *workspace, size_t workspace_size,
    void *out,
    const void *q,
    const void *k,
    const void *v,
    const void *total_kv_len,
    cudaStream_t stream,
    int max_threads_per_block) {

    return flash_attention_cuda::flash_attention_forward<float, float>(
        reinterpret_cast<float *>(out),
        reinterpret_cast<const float *>(q),
        reinterpret_cast<const float *>(k),
        reinterpret_cast<const float *>(v),
        reinterpret_cast<const int64_t *>(total_kv_len),
        info.batch_size,
        info.num_heads,
        info.num_kv_heads,
        info.seq_len_q,
        info.seq_len_kv,
        info.head_dim,
        info.scale,
        info.is_causal,
        info.has_variable_kv_len,
        info.q_stride_batch, info.q_stride_head, info.q_stride_seq,
        info.k_stride_batch, info.k_stride_head, info.k_stride_seq,
        info.v_stride_batch, info.v_stride_head, info.v_stride_seq,
        info.out_stride_batch, info.out_stride_head, info.out_stride_seq,
        reinterpret_cast<float *>(workspace),
        workspace_size / sizeof(float),
        stream);
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *out,
    const void *q,
    const void *k,
    const void *v,
    const void *total_kv_len,
    void *stream) const {

    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    int max_threads = _opaque->internal->maxThreadsPerBlock();

    // Dispatch based on data type
    if (_info.dtype == INFINI_DTYPE_F16) {
        return launchFlashAttentionKernel<half, float>(
            _info, workspace, workspace_size, out, q, k, v, total_kv_len,
            cuda_stream, max_threads);
    } else if (_info.dtype == INFINI_DTYPE_BF16) {
        return launchFlashAttentionKernel<__nv_bfloat16, float>(
            _info, workspace, workspace_size, out, q, k, v, total_kv_len,
            cuda_stream, max_threads);
    } else if (_info.dtype == INFINI_DTYPE_F32) {
        return launchFlashAttentionKernel<float, float>(
            _info, workspace, workspace_size, out, q, k, v, total_kv_len,
            cuda_stream, max_threads);
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::flash_attention::nvidia
