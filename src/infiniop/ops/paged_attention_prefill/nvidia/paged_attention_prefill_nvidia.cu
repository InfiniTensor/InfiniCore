#include <cuda_fp16.h>
#include <float.h>
#include <math.h>
#include <stdint.h>

#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../cuda/kernel.cuh"
#include "paged_attention_prefill_nvidia.cuh"

template <typename Tdata, typename Tcompute>
infiniStatus_t launchPagedAttentionPrefill(
    Tdata *out, const Tdata *q, const Tdata *k_cache, const Tdata *v_cache,
    const int64_t *block_tables,
    const int64_t *seq_lens,
    const int64_t *cum_seq_lens_q,
    const float *alibi_slopes,
    const size_t num_heads,
    const size_t num_seqs,
    const size_t num_kv_heads,
    const float scale,
    const size_t max_num_blocks_per_seq,
    const size_t block_size,
    const size_t total_q_tokens,
    const size_t head_size,
    const ptrdiff_t kv_block_stride,
    const ptrdiff_t kv_head_stride,
    const ptrdiff_t q_stride,
    const ptrdiff_t q_head_stride,
    cudaStream_t stream) {

    if (total_q_tokens == 0 || num_heads == 0) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    dim3 grid(total_q_tokens, num_heads);
    dim3 block(head_size);

    op::paged_attention_prefill::cuda::pagedAttentionPrefillKernel<Tdata, Tcompute>
        <<<grid, block, 0, stream>>>(
            out, q, k_cache, v_cache,
            block_tables, seq_lens, cum_seq_lens_q, alibi_slopes,
            num_heads, num_kv_heads, scale,
            max_num_blocks_per_seq, block_size,
            kv_block_stride, kv_head_stride,
            q_stride, q_head_stride,
            head_size,
            num_seqs);

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
    infiniopTensorDescriptor_t cum_seq_lens_q_desc,
    const std::optional<infiniopTensorDescriptor_t> &alibi_slopes_desc,
    float scale) {

    auto info = PagedAttentionPrefillInfo::create(
        out_desc, q_desc, k_cache_desc, v_cache_desc,
        block_tables_desc, seq_lens_desc,
        cum_seq_lens_q_desc,
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
    const void *block_tables,
    const void *seq_lens,
    const void *cum_seq_lens_q,
    const void *alibi_slopes,
    void *stream_) const {

    cudaStream_t stream = (cudaStream_t)stream_;

#define LAUNCH_KERNEL(Tdata, Tcompute)                                                             \
    launchPagedAttentionPrefill<Tdata, Tcompute>(                                                  \
        (Tdata *)out, (const Tdata *)q, (const Tdata *)k_cache, (const Tdata *)v_cache,            \
        (const int64_t *)block_tables, (const int64_t *)seq_lens, (const int64_t *)cum_seq_lens_q, \
        (const float *)alibi_slopes,                                                               \
        _info.num_heads, _info.num_seqs, _info.num_kv_heads,                                       \
        _info.scale, _info.max_num_blocks_per_seq,                                                 \
        _info.block_size, _info.total_q_tokens,                                                    \
        _info.head_size,                                                                           \
        _info.kv_block_stride, _info.kv_head_stride,                                               \
        _info.q_stride, _info.q_head_stride,                                                       \
        stream)

    if (_info.dtype == INFINI_DTYPE_F16) {
        return LAUNCH_KERNEL(half, float);
    } else if (_info.dtype == INFINI_DTYPE_BF16) {
        return LAUNCH_KERNEL(__nv_bfloat16, float);
    } else if (_info.dtype == INFINI_DTYPE_F32) {
        return LAUNCH_KERNEL(float, float);
    }

    return INFINI_STATUS_BAD_TENSOR_DTYPE;
}

} // namespace op::paged_attention_prefill::nvidia
