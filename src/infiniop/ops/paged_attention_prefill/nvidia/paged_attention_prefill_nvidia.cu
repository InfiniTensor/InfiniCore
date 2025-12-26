#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>
#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../cuda/kernel.cuh"
#include "paged_attention_prefill_nvidia.cuh"

template <typename Tdata, typename Tcompute, size_t HEAD_SIZE, size_t NUM_THREADS>
__global__ void pagedAttentionPrefillKernelLauncher(
    Tdata *out, const Tdata *q, const Tdata *k_cache, const Tdata *v_cache,
    const int32_t *block_tables, const int32_t *seq_lens, const float *alibi_slopes,
    const size_t num_kv_heads, const float scale, const size_t max_num_blocks_per_seq,
    const size_t block_size, const size_t max_new_len,
    const ptrdiff_t q_stride, const ptrdiff_t kv_block_stride, const ptrdiff_t kv_head_stride,
    const ptrdiff_t o_stride) {
    
    op::paged_attention_prefill::cuda::pagedAttentionPrefillKernel<Tdata, Tcompute, HEAD_SIZE, NUM_THREADS>(
        out, q, k_cache, v_cache, block_tables, seq_lens, alibi_slopes, num_kv_heads, 
        scale, max_num_blocks_per_seq, block_size, max_new_len, 
        q_stride, kv_block_stride, kv_head_stride, o_stride);
}

namespace op::paged_attention_prefill::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() { delete _opaque; }

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle, Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc, infiniopTensorDescriptor_t q_desc,
    infiniopTensorDescriptor_t k_cache_desc, infiniopTensorDescriptor_t v_cache_desc,
    infiniopTensorDescriptor_t block_tables_desc, infiniopTensorDescriptor_t seq_lens_desc,
    const std::optional<infiniopTensorDescriptor_t> &alibi_slopes_desc, float scale) {
    
    auto info = PagedAttentionPrefillInfo::create(out_desc, q_desc, k_cache_desc, v_cache_desc, 
                                                  block_tables_desc, seq_lens_desc, alibi_slopes_desc, scale);
    CHECK_RESULT(info);

    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        info.take(), 0, handle->device, handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size, void *out, const void *q, 
    const void *k_cache, const void *v_cache, const void *block_tables, 
    const void *seq_lens, const void *alibi_slopes, void *stream_) const {
    
    cudaStream_t stream = (cudaStream_t)stream_;
    dim3 grid(_info.num_heads, _info.num_seqs);
    dim3 block(512); // 使用 512 线程以获得更好的 occupancy

    // 共享内存计算：HEAD_SIZE 存储 Q + 最大的序列长度存储 Logits
    size_t max_total_len = _info.max_num_blocks_per_seq * _info.block_size;
    size_t shared_mem_size = (_info.head_size + max_total_len) * sizeof(float);

    if (_info.dtype == INFINI_DTYPE_F16) {
        pagedAttentionPrefillKernelLauncher<half, float, 128, 512><< <grid, block, shared_mem_size, stream >> >(
            (half *)out, (const half *)q, (const half *)k_cache, (const half *)v_cache,
            (const int32_t *)block_tables, (const int32_t *)seq_lens, (const float *)alibi_slopes,
            _info.num_kv_heads, _info.scale, _info.max_num_blocks_per_seq, _info.block_size, _info.max_new_len,
            _info.q_stride, _info.kv_block_stride, _info.kv_head_stride, _info.o_stride);
    } else if (_info.dtype == INFINI_DTYPE_BF16) {
        pagedAttentionPrefillKernelLauncher<__nv_bfloat16, float, 128, 512><< <grid, block, shared_mem_size, stream >> >(
            (__nv_bfloat16 *)out, (const __nv_bfloat16 *)q, (const __nv_bfloat16 *)k_cache, (const __nv_bfloat16 *)v_cache,
            (const int32_t *)block_tables, (const int32_t *)seq_lens, (const float *)alibi_slopes,
            _info.num_kv_heads, _info.scale, _info.max_num_blocks_per_seq, _info.block_size, _info.max_new_len,
            _info.q_stride, _info.kv_block_stride, _info.kv_head_stride, _info.o_stride);
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::paged_attention_prefill::nvidia
