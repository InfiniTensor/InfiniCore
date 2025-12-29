#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../cuda/kernel.cuh"
#include "paged_attention_prefill_nvidia.cuh"

// Host wrapper to launch the global kernel
// Removed INFINIOP_CUDA_KERNEL macro because this is a host function
template <typename Tdata, typename Tcompute>
infiniStatus_t launchPagedAttentionPrefill(
    Tdata *out, const Tdata *q, const Tdata *k_cache, const Tdata *v_cache,
    const int32_t *block_tables, const int32_t *seq_lens, const int32_t *new_lens,
    const float *alibi_slopes,
    const size_t num_heads, const size_t num_seqs, const size_t num_kv_heads, // Added num_seqs
    const float scale,
    const size_t max_num_blocks_per_seq, const size_t block_size, const size_t max_new_len,
    const ptrdiff_t q_stride, const ptrdiff_t kv_block_stride,
    const ptrdiff_t kv_head_stride, const ptrdiff_t o_stride,
    const size_t head_size,
    cudaStream_t stream) { // Added stream

    // Grid: [Head, Token (New), Sequence]
    dim3 grid(num_heads, max_new_len, num_seqs);
    // Block: [Head Size]
    dim3 block(head_size);

    size_t shared_mem_size = 0;

    op::paged_attention_prefill::cuda::pagedAttentionPrefillKernel<Tdata, Tcompute>
        <<<grid, block, shared_mem_size, stream>>>(
            out, q, k_cache, v_cache, block_tables, seq_lens, new_lens, alibi_slopes,
            num_heads, num_kv_heads, scale, max_num_blocks_per_seq, block_size, max_new_len,
            q_stride, kv_block_stride, kv_head_stride, o_stride, head_size);

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return INFINI_STATUS_INTERNAL_ERROR;
    }
    return INFINI_STATUS_SUCCESS;
}

namespace op::paged_attention_prefill::nvidia {

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
    infiniopTensorDescriptor_t k_cache_desc,
    infiniopTensorDescriptor_t v_cache_desc,
    infiniopTensorDescriptor_t block_tables_desc,
    infiniopTensorDescriptor_t seq_lens_desc,
    infiniopTensorDescriptor_t new_lens_desc,
    const std::optional<infiniopTensorDescriptor_t> &alibi_slopes_desc,
    float scale) {

    auto info = PagedAttentionPrefillInfo::create(out_desc, q_desc, k_cache_desc, v_cache_desc,
                                                  block_tables_desc, seq_lens_desc, new_lens_desc,
                                                  alibi_slopes_desc, scale);
    CHECK_RESULT(info);

    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        info.take(), 0, handle->device, handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *out, const void *q, const void *k_cache, const void *v_cache,
    const void *block_tables, const void *seq_lens, const void *new_lens,
    const void *alibi_slopes,
    void *stream_) const {

    cudaStream_t stream = (cudaStream_t)stream_;

    // Safety check
    if (_info.head_size > 1024) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    if (_info.dtype == INFINI_DTYPE_F16) {
        return launchPagedAttentionPrefill<half, float>(
            (half *)out, (const half *)q, (const half *)k_cache, (const half *)v_cache,
            (const int32_t *)block_tables, (const int32_t *)seq_lens, (const int32_t *)new_lens,
            (const float *)alibi_slopes,
            _info.num_heads, _info.num_seqs, _info.num_kv_heads, // Passed num_seqs
            _info.scale, _info.max_num_blocks_per_seq,
            _info.block_size, _info.max_new_len,
            _info.q_stride, _info.kv_block_stride, _info.kv_head_stride, _info.o_stride,
            _info.head_size,
            stream); // Passed stream
    } else if (_info.dtype == INFINI_DTYPE_BF16) {
        return launchPagedAttentionPrefill<__nv_bfloat16, float>(
            (__nv_bfloat16 *)out, (const __nv_bfloat16 *)q, (const __nv_bfloat16 *)k_cache, (const __nv_bfloat16 *)v_cache,
            (const int32_t *)block_tables, (const int32_t *)seq_lens, (const int32_t *)new_lens,
            (const float *)alibi_slopes,
            _info.num_heads, _info.num_seqs, _info.num_kv_heads,
            _info.scale, _info.max_num_blocks_per_seq,
            _info.block_size, _info.max_new_len,
            _info.q_stride, _info.kv_block_stride, _info.kv_head_stride, _info.o_stride,
            _info.head_size,
            stream);
    } else if (_info.dtype == INFINI_DTYPE_F32) {
        return launchPagedAttentionPrefill<float, float>(
            (float *)out, (const float *)q, (const float *)k_cache, (const float *)v_cache,
            (const int32_t *)block_tables, (const int32_t *)seq_lens, (const int32_t *)new_lens,
            (const float *)alibi_slopes,
            _info.num_heads, _info.num_seqs, _info.num_kv_heads,
            _info.scale, _info.max_num_blocks_per_seq,
            _info.block_size, _info.max_new_len,
            _info.q_stride, _info.kv_block_stride, _info.kv_head_stride, _info.o_stride,
            _info.head_size,
            stream);
    }

    return INFINI_STATUS_BAD_TENSOR_DTYPE;
}

} // namespace op::paged_attention_prefill::nvidia
