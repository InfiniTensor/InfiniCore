#ifndef __PAGED_ATTENTION_PREFILL_FP8_KERNEL_CUH__
#define __PAGED_ATTENTION_PREFILL_FP8_KERNEL_CUH__

//================================================================================
// FP8(E4M3) KV-cache gather-dequant kernels for paged prefill (plan B).
//
// The existing F16/BF16 prefill kernels stay untouched. For F8 caches we
// first gather the blocks referenced by each sequence's block table and
// dequantize them (e4m3_decode(code) * scale[block, kv_head, slot]) into a
// compact BF16/F16 scratch cache laid out as
//   [num_seqs * max_num_blocks_per_seq, num_kv_heads, page_block_size, head_size]
// together with identity block tables (scratch block of (seq, page) is
// seq * max_num_blocks_per_seq + page), then run the regular prefill kernel
// on the scratch. Scratch memory comes from the operator workspace.
//================================================================================

#include <type_traits>

#include "../../../devices/nvidia/nvidia_kernel_common.cuh"

namespace op::paged_attention_prefill::cuda {

namespace fp8 {

template <typename Tdata>
__device__ __forceinline__ Tdata fromFloat(float value) {
    if constexpr (std::is_same_v<Tdata, half>) {
        return __float2half_rn(value);
    } else if constexpr (std::is_same_v<Tdata, __nv_bfloat16>) {
        return __float2bfloat16_rn(value);
    } else {
        return static_cast<Tdata>(value);
    }
}

} // namespace fp8

// block_tables_scratch[i] = i for i in [0, total)
template <typename Tindex>
__device__ void fillIdentityBlockTablesKernel(
    Tindex *block_tables_scratch,
    const size_t total) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < total) {
        block_tables_scratch[i] = static_cast<Tindex>(i);
    }
}

// One CTA per (logical page, sequence, kv head): dequantize the referenced
// page of K and V into the compact scratch cache. Pages past the sequence's
// kv length are skipped.
template <typename Tindex, typename Tdata>
__device__ void gatherDequantFp8KvKernel(
    Tdata *k_scratch_,
    Tdata *v_scratch_,
    const uint8_t *k_cache_,
    const uint8_t *v_cache_,
    const float *k_scale_,
    const float *v_scale_,
    const Tindex *block_tables_,
    const Tindex *total_kv_lens_,
    const size_t num_kv_heads,
    const size_t head_size,
    const size_t value_size,
    const size_t page_block_size,
    const size_t max_num_blocks_per_seq,
    const ptrdiff_t block_table_batch_stride,
    const ptrdiff_t k_batch_stride,
    const ptrdiff_t k_row_stride,
    const ptrdiff_t k_head_stride,
    const ptrdiff_t v_batch_stride,
    const ptrdiff_t v_row_stride,
    const ptrdiff_t v_head_stride,
    const ptrdiff_t k_scale_block_stride,
    const ptrdiff_t k_scale_head_stride,
    const ptrdiff_t k_scale_slot_stride,
    const ptrdiff_t v_scale_block_stride,
    const ptrdiff_t v_scale_head_stride,
    const ptrdiff_t v_scale_slot_stride) {

    const size_t page = blockIdx.x;
    const size_t seq = blockIdx.y;
    const size_t head = blockIdx.z;

    const size_t kv_len = static_cast<size_t>(total_kv_lens_[seq]);
    if (page * page_block_size >= kv_len) {
        return; // page not referenced by this sequence
    }

    const ptrdiff_t phys = static_cast<ptrdiff_t>(block_tables_[seq * block_table_batch_stride + page]);
    const size_t scratch_block = seq * max_num_blocks_per_seq + page;

    // K page
    {
        const uint8_t *k_src = k_cache_ + phys * k_batch_stride + static_cast<ptrdiff_t>(head) * k_head_stride;
        const float *k_scl = k_scale_ + phys * k_scale_block_stride + static_cast<ptrdiff_t>(head) * k_scale_head_stride;
        Tdata *k_dst = k_scratch_ + (scratch_block * num_kv_heads + head) * (page_block_size * head_size);
        for (size_t idx = threadIdx.x; idx < page_block_size * head_size; idx += blockDim.x) {
            const size_t slot = idx / head_size;
            const size_t d = idx - slot * head_size;
            k_dst[idx] = fp8::fromFloat<Tdata>(
                infiniopFp8E4m3Decode(k_src[slot * k_row_stride + d]) * k_scl[slot * k_scale_slot_stride]);
        }
    }

    // V page (value_size may differ from head_size for MLA)
    {
        const uint8_t *v_src = v_cache_ + phys * v_batch_stride + static_cast<ptrdiff_t>(head) * v_head_stride;
        const float *v_scl = v_scale_ + phys * v_scale_block_stride + static_cast<ptrdiff_t>(head) * v_scale_head_stride;
        Tdata *v_dst = v_scratch_ + (scratch_block * num_kv_heads + head) * (page_block_size * value_size);
        for (size_t idx = threadIdx.x; idx < page_block_size * value_size; idx += blockDim.x) {
            const size_t slot = idx / value_size;
            const size_t d = idx - slot * value_size;
            v_dst[idx] = fp8::fromFloat<Tdata>(
                infiniopFp8E4m3Decode(v_src[slot * v_row_stride + d]) * v_scl[slot * v_scale_slot_stride]);
        }
    }
}

} // namespace op::paged_attention_prefill::cuda

#endif // __PAGED_ATTENTION_PREFILL_FP8_KERNEL_CUH__
