#include <cuda_fp16.h>
#include <float.h>
#include <math.h>
#include <stdint.h>

#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../cuda/kernel.cuh"
#include "paged_attention_prefill_nvidia.cuh"

// ==============================================================================
// Host wrapper to launch the global kernel
// ==============================================================================
template <typename Tdata, typename Tcompute>
infiniStatus_t launchPagedAttentionPrefill(
    Tdata *out, const Tdata *q, const Tdata *k_cache, const Tdata *v_cache,
    const int64_t *block_tables, const int64_t *cache_lens, const int64_t *seq_lens,
    const int64_t *offset,
    const float *alibi_slopes,
    const size_t num_heads,
    const size_t num_seqs,
    const size_t num_kv_heads,
    const float scale,
    const size_t max_num_blocks_per_seq,
    const size_t block_size,
    const size_t total_q_tokens,
    const ptrdiff_t q_stride,
    const ptrdiff_t kv_block_stride,
    const ptrdiff_t kv_head_stride,
    const ptrdiff_t o_stride,
    const size_t head_size,
    cudaStream_t stream) {

    if (total_q_tokens == 0 || num_heads == 0) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    // 使用 2D Grid: X轴是所有 Token，Y轴是所有 Head
    dim3 grid(total_q_tokens, num_heads);
    dim3 block(head_size);

    op::paged_attention_prefill::cuda::pagedAttentionPrefillKernel<Tdata, Tcompute>
        <<<grid, block, 0, stream>>>(
            out, q, k_cache, v_cache,
            block_tables, cache_lens, seq_lens, alibi_slopes,
            num_heads, num_kv_heads, scale,
            max_num_blocks_per_seq, block_size,
            kv_block_stride, kv_head_stride,
            head_size,
            offset, num_seqs);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Kernel Launch Failed: " << cudaGetErrorString(err) << std::endl;
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
    infiniopTensorDescriptor_t cache_lens_desc,
    infiniopTensorDescriptor_t seq_lens_desc,
    infiniopTensorDescriptor_t offset_desc,
    const std::optional<infiniopTensorDescriptor_t> &alibi_slopes_desc,
    float scale) {

    auto info = PagedAttentionPrefillInfo::create(out_desc, q_desc, k_cache_desc, v_cache_desc,
                                                  block_tables_desc, cache_lens_desc, seq_lens_desc,
                                                  offset_desc,
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
    const void *block_tables, const void *cache_lens, const void *seq_lens,
    const void *offset,
    const void *alibi_slopes,
    void *stream_) const {

    cudaStream_t stream = (cudaStream_t)stream_;

    if (_info.head_size > 1024) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

#define LAUNCH_KERNEL(Tdata, Tcompute)                                                         \
    launchPagedAttentionPrefill<Tdata, Tcompute>(                                              \
        (Tdata *)out, (const Tdata *)q, (const Tdata *)k_cache, (const Tdata *)v_cache,        \
        (const int64_t *)block_tables, (const int64_t *)cache_lens, (const int64_t *)seq_lens, \
        (const int64_t *)offset,                                                               \
        (const float *)alibi_slopes,                                                           \
        _info.num_heads, _info.num_seqs, _info.num_kv_heads,                                   \
        _info.scale, _info.max_num_blocks_per_seq,                                             \
        _info.block_size, _info.total_q_tokens,                                                \
        _info.q_stride, _info.kv_block_stride, _info.kv_head_stride, _info.o_stride,           \
        _info.head_size,                                                                       \
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
