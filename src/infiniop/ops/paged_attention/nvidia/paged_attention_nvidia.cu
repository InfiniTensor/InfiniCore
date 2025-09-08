#include "../../../devices/nvidia/nvidia_common.cuh"
#include "paged_attention_nvidia.cuh"

#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include <cub/block/block_reduce.cuh>
#include "../../../reduce/cuda/reduce.cuh"

#include "../cuda/kernel.cuh"
#include <stdio.h>
#include <cuda_runtime.h>

template <typename Tdata, typename Tcompute, size_t HEAD_SIZE, size_t NUM_THREADS>
INFINIOP_CUDA_KERNEL pagedAttention(
    Tdata *out, const Tdata *q, const Tdata *k_cache, const Tdata *v_cache,
    const int32_t *block_tables, const int32_t *seq_lens, const float *alibi_slopes,
    const size_t num_kv_heads, const float scale, const size_t max_num_blocks_per_seq,
    const size_t block_size,
    const ptrdiff_t q_stride, const ptrdiff_t kv_block_stride, const ptrdiff_t kv_head_stride
    ) {
    op::paged_attention::cuda::pagedAttentionKernel<Tdata, Tcompute, HEAD_SIZE, NUM_THREADS>(
        out, q, k_cache, v_cache, block_tables, seq_lens, alibi_slopes, num_kv_heads, scale, 
        max_num_blocks_per_seq, block_size, q_stride, kv_block_stride, kv_head_stride);
}

namespace op::paged_attention::nvidia {

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
    const std::optional<infiniopTensorDescriptor_t>& alibi_slopes_desc,
    float scale
    ) {
    auto info = PagedAttentionInfo::create(out_desc, q_desc, k_cache_desc, v_cache_desc, block_tables_desc, seq_lens_desc, alibi_slopes_desc, scale);
    CHECK_RESULT(info);
    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        info.take(), 0, handle->device, handle->device_id);
    
    return INFINI_STATUS_SUCCESS;
}

template <size_t HEAD_SIZE, size_t NUM_THREADS>
infiniStatus_t launchKernel(void *out, const void *q, const void *k_cache, const void *v_cache,
                    infiniDtype_t dtype,
                    const void *block_tables, const void *seq_lens, const void *alibi_slopes,
                    size_t num_heads, size_t num_seqs,
                    size_t num_kv_heads, float scale, size_t max_num_blocks_per_seq, size_t block_size,
                    ptrdiff_t q_stride, ptrdiff_t kv_block_stride, ptrdiff_t kv_head_stride,
                    cudaStream_t stream) {
    dim3 grid(uint32_t(num_heads), uint32_t(num_seqs), 1);
    dim3 block(NUM_THREADS);
    size_t shared_mem_size = (HEAD_SIZE + max_num_blocks_per_seq * block_size + 2) * sizeof(float);

    // size_t shared_mem_size = 16;
    if (dtype == INFINI_DTYPE_F16) {
        // size_t shared_mem_size = (HEAD_SIZE + max_num_blocks_per_seq * block_size + 2) * sizeof(Tcompute);
        pagedAttention<half, float, HEAD_SIZE, NUM_THREADS>
            <<<grid, block, shared_mem_size, stream>>>(
                (half*)out,
                (const half*)q, (const half*)k_cache, (const half*)v_cache, 
                (const int32_t*)block_tables, (const int32_t*)seq_lens, (const float*)alibi_slopes, num_kv_heads,
                scale, max_num_blocks_per_seq, block_size,
                q_stride, kv_block_stride, kv_head_stride
            );
    } else if (dtype == INFINI_DTYPE_BF16) {
        // size_t shared_mem_size = (HEAD_SIZE + max_num_blocks_per_seq * block_size + 2) * sizeof(float);
        pagedAttention<__nv_bfloat16, float, HEAD_SIZE, NUM_THREADS>
            <<<grid, block, shared_mem_size, stream>>>(
                (__nv_bfloat16*)out, (const __nv_bfloat16*)q, (const __nv_bfloat16*)k_cache, (const __nv_bfloat16*)v_cache, 
                (const int32_t*)block_tables, (const int32_t*)seq_lens, (const float*)alibi_slopes, num_kv_heads,
                scale, max_num_blocks_per_seq, block_size,
                q_stride, kv_block_stride, kv_head_stride
            );
    } else if (dtype == INFINI_DTYPE_F32) {
        // size_t shared_mem_size = (HEAD_SIZE + max_num_blocks_per_seq * block_size + 2) * sizeof(float);
        pagedAttention<float, float, HEAD_SIZE, NUM_THREADS>
            <<<grid, block, shared_mem_size, stream>>>(
                (float*)out, (const float*)q, (const float*)k_cache, (const float*)v_cache, 
                (const int32_t*)block_tables, (const int32_t*)seq_lens, (const float*)alibi_slopes, num_kv_heads,
                scale, max_num_blocks_per_seq, block_size,
                q_stride, kv_block_stride, kv_head_stride
            );
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *out, const void *q, const void *k_cache, const void *v_cache,
    const void *block_tables, const void *seq_lens, const void *alibi_slopes,
    void *stream_) const {
    cudaStream_t stream = (cudaStream_t)stream_;
    if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_1024) {
        if (_info.head_size == 128) {
            launchKernel<128, CUDA_BLOCK_SIZE_1024>(
                    out, q, k_cache, v_cache, _info.dtype, block_tables, seq_lens, alibi_slopes, 
                    _info.num_heads, _info.num_seqs,
                    _info.num_kv_heads, _info.scale, _info.max_num_blocks_per_seq, _info.block_size, 
                    _info.q_stride, _info.kv_block_stride, _info.kv_head_stride,
                    stream);
            
        } 
        // else if head_size=128, block_size=16）for llama
        else {
            printf("head_size: %zu\n", _info.head_size);
            return INFINI_STATUS_BAD_TENSOR_SHAPE; 
        }
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_512) {
        if (_info.head_size == 128 ) {
            launchKernel<128, CUDA_BLOCK_SIZE_512>(
                    out, q, k_cache, v_cache, _info.dtype, block_tables, seq_lens, alibi_slopes, 
                    _info.num_heads, _info.num_seqs,
                    _info.num_kv_heads, _info.scale, _info.max_num_blocks_per_seq, _info.block_size, 
                    _info.q_stride, _info.kv_block_stride, _info.kv_head_stride,
                    stream);
            
        } 
        // else if head_size=128, block_size=16）for llama
        else {
            printf("head_size: %zu\n", _info.head_size);
            return INFINI_STATUS_BAD_TENSOR_SHAPE; 
        }
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_4096) {
        if (_info.head_size == 128 ) {
            launchKernel<128, CUDA_BLOCK_SIZE_4096>(
                    out, q, k_cache, v_cache, _info.dtype, block_tables, seq_lens, alibi_slopes, 
                    _info.num_heads, _info.num_seqs,
                    _info.num_kv_heads, _info.scale, _info.max_num_blocks_per_seq, _info.block_size, 
                    _info.q_stride, _info.kv_block_stride, _info.kv_head_stride,
                    stream);
        } 
        // else if head_size=128, block_size=16）for llama
        else {
            printf("head_size: %zu", _info.head_size);
            return INFINI_STATUS_BAD_TENSOR_SHAPE; 
        }
    } else {
        return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
    }
    
    
    return INFINI_STATUS_SUCCESS;
}

} 