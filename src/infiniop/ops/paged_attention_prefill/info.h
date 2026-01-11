#ifndef __INFINIOP_PAGED_ATTENTION_PREFILL_INFO_H__
#define __INFINIOP_PAGED_ATTENTION_PREFILL_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <iostream>
#include <optional>
#include <vector>

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
    size_t total_q_tokens;

    ptrdiff_t q_stride;
    ptrdiff_t q_head_stride;
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
        infiniopTensorDescriptor_t cum_seq_lens_q_desc,
        const std::optional<infiniopTensorDescriptor_t> &alibi_slopes_desc,
        float scale) {

        auto dtype = q_desc->dtype();
        CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32);

        if (out_desc->dtype() != dtype || k_cache_desc->dtype() != dtype || v_cache_desc->dtype() != dtype) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        if (cum_seq_lens_q_desc->dtype() != INFINI_DTYPE_I64 || seq_lens_desc->dtype() != INFINI_DTYPE_I64) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        if (alibi_slopes_desc.has_value() && alibi_slopes_desc.value() != nullptr) {
        }

        auto k_shape = k_cache_desc->shape();
        auto v_shape = v_cache_desc->shape();
        auto block_tables_shape = block_tables_desc->shape();
        auto seq_lens_shape = seq_lens_desc->shape();
        auto cum_seq_lens_q_shape = cum_seq_lens_q_desc->shape();

        if (k_shape.size() != 4 || v_shape.size() != 4) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        if (block_tables_shape.size() != 2) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        if (seq_lens_shape.size() != 1 || cum_seq_lens_q_shape.size() != 1) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        if (cum_seq_lens_q_shape[0] != seq_lens_shape[0] + 1) {
            return INFINI_STATUS_BAD_PARAM;
        }

        // Q shape: [total_tokens, heads, dim]
        auto q_shape = q_desc->shape();
        if (q_shape.size() != 3) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        size_t total_q_tokens = q_shape[0];
        size_t num_heads = q_shape[1];
        size_t head_size = q_shape[2];

        if (head_size > 1024) {
            return INFINI_STATUS_BAD_PARAM;
        }

        size_t num_seqs = seq_lens_shape[0];
        size_t num_kv_heads = k_shape[1];
        size_t block_size = k_shape[2];
        size_t max_num_blocks_per_seq = block_tables_shape[1];

        ptrdiff_t q_stride = q_desc->stride(0);
        ptrdiff_t q_head_stride = q_desc->stride(1);
        ptrdiff_t kv_block_stride = k_cache_desc->stride(0);
        ptrdiff_t kv_head_stride = k_cache_desc->stride(1);
        ptrdiff_t o_stride = out_desc->stride(0);

        return utils::Result<PagedAttentionPrefillInfo>(PagedAttentionPrefillInfo{
            dtype,
            scale,
            num_seqs,
            num_heads,
            num_kv_heads,
            head_size,
            block_size,
            max_num_blocks_per_seq,
            total_q_tokens,
            q_stride,
            q_head_stride,
            kv_block_stride,
            kv_head_stride,
            o_stride});
    }
};

} // namespace op::paged_attention_prefill

#endif
