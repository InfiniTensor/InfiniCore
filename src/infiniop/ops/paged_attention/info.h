#ifndef __PAGED_ATTENTION_INFO_H__
#define __PAGED_ATTENTION_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <vector>
#include <optional>

namespace op::paged_attention {

class PagedAttentionInfo {
    PagedAttentionInfo() = default;

public:
    // --- Data Types and Scale ---
    infiniDtype_t dtype;
    float scale;

    // --- Shape Dimensions ---
    size_t num_seqs;
    size_t num_heads;
    size_t num_kv_heads;
    size_t head_size;
    size_t block_size;
    size_t max_num_blocks_per_seq;
    // size_t max_seq_len;

    // --- Strides for Memory Layout ---
    ptrdiff_t q_stride;           
    ptrdiff_t kv_block_stride;    
    ptrdiff_t kv_head_stride;     


    static utils::Result<PagedAttentionInfo> create(
        infiniopTensorDescriptor_t out_desc,
        infiniopTensorDescriptor_t q_desc,
        infiniopTensorDescriptor_t k_cache_desc,
        infiniopTensorDescriptor_t v_cache_desc,
        infiniopTensorDescriptor_t block_tables_desc,
        infiniopTensorDescriptor_t seq_lens_desc,
        const std::optional<infiniopTensorDescriptor_t>& alibi_slopes_desc,
        float scale) {
        
        auto dtype = q_desc->dtype();
        CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32); 
        if (out_desc->dtype() != dtype || k_cache_desc->dtype() != dtype || v_cache_desc->dtype() != dtype) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        if (q_desc->ndim() != 3 || k_cache_desc->ndim() < 4 || v_cache_desc->ndim() < 4 || 
            block_tables_desc->ndim() != 2 || seq_lens_desc->ndim() != 1) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        
        // --- Extract shape dimensions ---
        auto q_shape = q_desc->shape();
        auto k_cache_shape = k_cache_desc->shape();
        // k_cache_shape: [num_blocks, num_kv_heads, block_size, head_size]
        
        size_t num_seqs = q_shape[0];
        size_t num_heads = q_shape[1];
        size_t head_size = q_shape[2];
        
        size_t num_kv_heads = k_cache_shape[1];
        size_t block_size = v_cache_desc->shape()[2]; // 使用V cache的block size维度更可靠
        size_t max_num_blocks_per_seq = block_tables_desc->shape()[1];

        // --- Calculate max_seq_len for shared memory allocation ---
        // This is a safe upper bound.
        // info.max_seq_len = info.max_num_blocks_per_seq * info.block_size;

        // --- Extract strides for memory access ---
        ptrdiff_t q_stride = q_desc->stride(0);
        ptrdiff_t kv_block_stride = k_cache_desc->stride(0);
        ptrdiff_t kv_head_stride = k_cache_desc->stride(1);
        // Note: We assume k_cache and v_cache have compatible strides.
        // A more robust implementation could check v_cache_desc->stride() as well.

        // --- Check for optional features ---
        // info.has_alibi = alibi_slopes_desc.has_value();

        return utils::Result<PagedAttentionInfo>(PagedAttentionInfo{
            dtype,
            scale,
            num_seqs,
            num_heads,
            num_kv_heads,
            head_size,
            block_size,
            max_num_blocks_per_seq,
            q_stride,
            kv_block_stride,
            kv_head_stride
        });
    }
};

} // namespace op::paged_attention

#endif // __PAGED_ATTENTION_INFO_H__