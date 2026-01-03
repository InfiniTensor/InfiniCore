#ifndef __PAGED_ATTENTION_PREFILL_INFO_H__
#define __PAGED_ATTENTION_PREFILL_INFO_H__

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
    ptrdiff_t kv_block_stride;
    ptrdiff_t kv_head_stride;
    ptrdiff_t o_stride;

    static utils::Result<PagedAttentionPrefillInfo> create(
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

        auto dtype = q_desc->dtype();
        CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32);

        if (out_desc->dtype() != dtype || k_cache_desc->dtype() != dtype || v_cache_desc->dtype() != dtype) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        if (offset_desc->dtype() != INFINI_DTYPE_I64 || seq_lens_desc->dtype() != INFINI_DTYPE_I64) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        if (alibi_slopes_desc.has_value() && alibi_slopes_desc.value() != nullptr) {
            std::cerr << "[Error] PagedAttentionPrefill: ALiBi slopes are not supported yet." << std::endl;
            return INFINI_STATUS_BAD_PARAM;
        }

        // Q shape: [total_tokens, heads, dim] (3D)
        auto q_shape = q_desc->shape();
        if (q_shape.size() < 3) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        size_t total_q_tokens = q_shape[0];

        size_t num_heads = q_shape[q_shape.size() - 2];
        size_t head_size = q_shape[q_shape.size() - 1];

        if (head_size != 128) {
            std::cerr << "[Error] PagedAttentionPrefill head_size = 128 supported, got " << head_size << std::endl;
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        // 从 seq_lens 获取 num_seqs
        size_t num_seqs = seq_lens_desc->shape()[0];

        auto k_cache_shape = k_cache_desc->shape();
        size_t num_kv_heads = k_cache_shape[1];
        size_t block_size = v_cache_desc->shape()[2];
        size_t max_num_blocks_per_seq = block_tables_desc->shape()[1];

        // 提取步长,需要保持多个请求的 Q 连续
        ptrdiff_t q_stride = q_desc->stride(0);
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
            kv_block_stride,
            kv_head_stride,
            o_stride});
    }
};

} // namespace op::paged_attention_prefill

#endif
