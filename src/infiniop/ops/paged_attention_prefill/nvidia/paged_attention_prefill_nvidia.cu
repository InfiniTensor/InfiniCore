

#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"

#include "../../../reduce/cuda/reduce.cuh"

#include "../cuda/kernel.cuh"
#include "paged_attention_prefill_nvidia.cuh"

template <typename Tdata,
          size_t HEAD_DIM,
          size_t NUM_THREADS_PER_BLOCK>
static inline infiniStatus_t launch_prefill_kernel_impl(
    void *out,
    const void *q,
    const void *k_cache,
    const void *v_cache,
    const void *block_tables,
    const void *seq_lens,
    const void *context_lens,
    const void *seq_offsets,
    size_t num_heads,
    size_t num_seqs,
    size_t num_kv_heads,
    size_t max_blocks_per_seq,
    size_t block_size,
    ptrdiff_t q_stride,
    ptrdiff_t kv_block_stride,
    ptrdiff_t kv_head_stride,
    ptrdiff_t out_stride,
    float scale,
    cudaStream_t stream) {
    dim3 grid(num_heads, num_seqs, 1);
    dim3 block(NUM_THREADS_PER_BLOCK);

    // Shared memory:
    // - K block: block_size * HEAD_DIM
    // - V block: block_size * HEAD_DIM
    size_t smem_bytes = 2 * block_size * HEAD_DIM * sizeof(Tdata);

    op::paged_attention_prefill::cuda::paged_attention_prefill<Tdata, Tdata, HEAD_DIM, NUM_THREADS_PER_BLOCK>
        <<<grid, block, smem_bytes, stream>>>(
            (Tdata *)out,
            (const Tdata *)q,
            (const Tdata *)k_cache,
            (const Tdata *)v_cache,
            (const int64_t *)block_tables,
            (const int64_t *)seq_lens,
            (const int64_t *)context_lens,
            (const int64_t *)seq_offsets,
            num_heads,
            num_kv_heads,
            max_blocks_per_seq,
            block_size,
            q_stride,
            kv_block_stride,
            kv_head_stride,
            out_stride,
            scale);

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
    infiniopTensorDescriptor_t seq_offsets_desc,
    infiniopTensorDescriptor_t cache_lens_desc,
    const std::optional<infiniopTensorDescriptor_t> &alibi_slopes_desc,
    float scale) {
    auto info = PagedAttentionPrefillInfo::create(out_desc, q_desc, k_cache_desc, v_cache_desc, block_tables_desc, seq_lens_desc, seq_offsets_desc, cache_lens_desc, alibi_slopes_desc, scale);
    CHECK_RESULT(info);
    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        info.take(), 0, handle->device, handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

template <size_t NUM_THREADS_PER_BLOCK>
static infiniStatus_t dispatch_prefill_head_dim_and_dtype(
    infiniDtype_t dtype,
    size_t head_dim,
    void *out,
    const void *q,
    const void *k_cache,
    const void *v_cache,
    const void *block_tables,
    const void *seq_lens,
    const void *context_lens,
    const void *seq_offsets,
    size_t num_heads,
    size_t num_seqs,
    size_t num_kv_heads,
    size_t max_blocks_per_seq,
    size_t block_size,
    ptrdiff_t q_stride,
    ptrdiff_t kv_block_stride,
    ptrdiff_t kv_head_stride,
    ptrdiff_t out_stride,
    float scale,
    cudaStream_t stream) {
#define DISPATCH_HEAD_DIM(HD)                                                            \
    case HD:                                                                             \
        if (dtype == INFINI_DTYPE_F16)                                                   \
            return launch_prefill_kernel_impl<half, HD, NUM_THREADS_PER_BLOCK>(          \
                out, q, k_cache, v_cache, block_tables,                                  \
                seq_lens, context_lens, seq_offsets,                                     \
                num_heads, num_seqs, num_kv_heads,                                       \
                max_blocks_per_seq, block_size,                                          \
                q_stride, kv_block_stride, kv_head_stride, out_stride,                   \
                scale, stream);                                                          \
        if (dtype == INFINI_DTYPE_BF16)                                                  \
            return launch_prefill_kernel_impl<__nv_bfloat16, HD, NUM_THREADS_PER_BLOCK>( \
                out, q, k_cache, v_cache, block_tables,                                  \
                seq_lens, context_lens, seq_offsets,                                     \
                num_heads, num_seqs, num_kv_heads,                                       \
                max_blocks_per_seq, block_size,                                          \
                q_stride, kv_block_stride, kv_head_stride, out_stride,                   \
                scale, stream);                                                          \
        if (dtype == INFINI_DTYPE_F32)                                                   \
            return launch_prefill_kernel_impl<float, HD, NUM_THREADS_PER_BLOCK>(         \
                out, q, k_cache, v_cache, block_tables,                                  \
                seq_lens, context_lens, seq_offsets,                                     \
                num_heads, num_seqs, num_kv_heads,                                       \
                max_blocks_per_seq, block_size,                                          \
                q_stride, kv_block_stride, kv_head_stride, out_stride,                   \
                scale, stream);                                                          \
        return INFINI_STATUS_BAD_TENSOR_DTYPE;

    switch (head_dim) {
        DISPATCH_HEAD_DIM(16)
        DISPATCH_HEAD_DIM(32)
        DISPATCH_HEAD_DIM(64)
        DISPATCH_HEAD_DIM(128)
        DISPATCH_HEAD_DIM(256)
    default:
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

#undef DISPATCH_HEAD_DIM
}

infiniStatus_t Descriptor::calculate(
    void *,
    size_t,
    void *out,
    const void *q,
    const void *k_cache,
    const void *v_cache,
    const void *block_tables,
    const void *seq_lens,
    const void *seq_offsets,
    const void *context_lens,
    const void *alibi_slopes,
    void *stream_) const {
    cudaStream_t stream = (cudaStream_t)stream_;
    const size_t max_threads = _opaque->internal->maxThreadsPerBlock();

    if (max_threads == CUDA_BLOCK_SIZE_4096) {
        return dispatch_prefill_head_dim_and_dtype<CUDA_BLOCK_SIZE_4096>(
            _info.dtype,
            _info.head_size,
            out, q, k_cache, v_cache,
            block_tables, seq_lens, context_lens, seq_offsets,
            _info.num_heads,
            _info.num_seqs,
            _info.num_kv_heads,
            _info.max_num_blocks_per_seq,
            _info.block_size,
            _info.q_stride,
            _info.kv_block_stride,
            _info.kv_head_stride,
            _info.o_stride,
            _info.scale,
            stream);
    } else if (max_threads == CUDA_BLOCK_SIZE_1024) {
        return dispatch_prefill_head_dim_and_dtype<CUDA_BLOCK_SIZE_1024>(
            _info.dtype,
            _info.head_size,
            out, q, k_cache, v_cache,
            block_tables, seq_lens, context_lens, seq_offsets,
            _info.num_heads,
            _info.num_seqs,
            _info.num_kv_heads,
            _info.max_num_blocks_per_seq,
            _info.block_size,
            _info.q_stride,
            _info.kv_block_stride,
            _info.kv_head_stride,
            _info.o_stride,
            _info.scale,
            stream);
    } else if (max_threads == CUDA_BLOCK_SIZE_512) {
        return dispatch_prefill_head_dim_and_dtype<CUDA_BLOCK_SIZE_512>(
            _info.dtype,
            _info.head_size,
            out, q, k_cache, v_cache,
            block_tables, seq_lens, context_lens, seq_offsets,
            _info.num_heads,
            _info.num_seqs,
            _info.num_kv_heads,
            _info.max_num_blocks_per_seq,
            _info.block_size,
            _info.q_stride,
            _info.kv_block_stride,
            _info.kv_head_stride,
            _info.o_stride,
            _info.scale,
            stream);
    }

    return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
}

} // namespace op::paged_attention_prefill::nvidia
