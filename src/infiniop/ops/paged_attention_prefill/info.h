#ifndef __PAGED_ATTENTION_PREFILL_INFO_H__
#define __PAGED_ATTENTION_PREFILL_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <iostream>
#include <optional>

namespace op::paged_attention_prefill {

class PagedAttentionPrefillInfo {
    PagedAttentionPrefillInfo() = default;

public:
    infiniDtype_t dtype;
    float scale;

    size_t num_seqs;
    size_t num_heads;
    size_t num_kv_heads;
    size_t head_size;
    size_t block_size;
    size_t max_num_blocks_per_seq;
    size_t max_new_len; // Prefill 特有的维度

    ptrdiff_t q_stride;
    ptrdiff_t kv_block_stride;
    ptrdiff_t kv_head_stride;
    ptrdiff_t o_stride;

    static utils::Result<PagedAttentionPrefillInfo> create(
        infiniopTensorDescriptor_t out_desc,
        infiniopTensorDescriptor_t q_desc,
        infiniopTensorDescriptor_t k_cache_desc,
        infiniopTensorDescriptor_t v_cache_desc,
        infiniopTensorDescriptor_t block_tables_desc,
        infiniopTensorDescriptor_t seq_lens_desc,
        const std::optional<infiniopTensorDescriptor_t> &alibi_slopes_desc,
        float scale) {

        auto dtype = q_desc->dtype();
        // 基本类型检查
        CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32);
        
        // 维度检查：Prefill 的 Q 可能是 4D [batch, max_new_len, n_heads, dh] 
        // 或者是 3D [total_new_tokens, n_heads, dh]
        if (q_desc->ndim() < 3 || block_tables_desc->ndim() != 2) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        auto q_shape = q_desc->shape();
        size_t num_seqs = q_shape[0];
        size_t max_new_len = q_shape[1]; 
        size_t num_heads = q_shape[2];
        size_t head_size = q_shape[3];

        size_t num_kv_heads = k_cache_desc->shape()[1];
        size_t block_size = v_cache_desc->shape()[2];
        size_t max_num_blocks_per_seq = block_tables_desc->shape()[1];

        return utils::Result<PagedAttentionPrefillInfo>(PagedAttentionPrefillInfo{
            dtype, scale, num_seqs, num_heads, num_kv_heads, head_size, 
            block_size, max_num_blocks_per_seq, max_new_len,
            q_desc->stride(0), k_cache_desc->stride(0), k_cache_desc->stride(1), out_desc->stride(0)});
    }
};

} // namespace op::paged_attention_prefill

#endif
