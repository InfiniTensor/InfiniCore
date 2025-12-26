#ifndef __PAGED_CACHING_PREFILL_KERNEL_CUH__
#define __PAGED_CACHING_PREFILL_KERNEL_CUH__

#include <cuda_runtime.h>
#include <stdint.h>

namespace op::paged_caching_prefill::cuda {

template <typename Tdata, int NUM_THREADS>
__device__ void pagedCachingPrefillKernel(
    Tdata *k_cache_ptr,
    Tdata *v_cache_ptr,
    const Tdata *k_ptr,
    const Tdata *v_ptr,
    const int32_t *slot_mapping_ptr,
    const size_t head_size,
    const size_t block_size,
    const ptrdiff_t k_src_stride,
    const ptrdiff_t v_src_stride,
    const ptrdiff_t k_cache_block_stride,
    const ptrdiff_t v_cache_block_stride
) {
    // grid: x = head_idx, y = token_idx
    const int head_idx = blockIdx.x;
    const int token_idx = blockIdx.y;

    // 获取该 Token 对应的全局物理槽位
    const int32_t slot_idx = slot_mapping_ptr[token_idx];
    if (slot_idx < 0) return;

    const int32_t physical_block_idx = slot_idx / block_size;
    const int32_t block_offset = slot_idx % block_size;

    // 源地址计算: [ntok, nkvh, dh]
    const Tdata *k_src = k_ptr + token_idx * k_src_stride + head_idx * head_size;
    const Tdata *v_src = v_ptr + token_idx * v_src_stride + head_idx * head_size;

    // 目标地址计算 (假设布局: [num_blocks, nkvh, block_size, dh])
    // 步长说明: head_stride = block_size * head_size
    const ptrdiff_t cache_head_stride = block_size * head_size;

    Tdata *k_dst = k_cache_ptr + physical_block_idx * k_cache_block_stride 
                               + head_idx * cache_head_stride 
                               + block_offset * head_size;
                               
    Tdata *v_dst = v_cache_ptr + physical_block_idx * v_cache_block_stride 
                               + head_idx * cache_head_stride 
                               + block_offset * head_size;

    // 元素拷贝
    for (int i = threadIdx.x; i < head_size; i += NUM_THREADS) {
        k_dst[i] = k_src[i];
        v_dst[i] = v_src[i];
    }
}

} // namespace op::paged_caching_prefill::cuda

#endif
