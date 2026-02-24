#ifndef __PAGED_ATTENTION_V1_INFO_H__
#define __PAGED_ATTENTION_V1_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <iostream>
#include <optional>
#include <vector>

namespace op::paged_attention_v1 {

class PagedAttentionV1Info {
    PagedAttentionV1Info() = default;

public:
    infiniDtype_t dtype;
    infiniDtype_t index_dtype;
    double scale;

    size_t num_seqs;
    size_t num_heads;
    size_t num_kv_heads;
    size_t head_size;
    size_t page_block_size;
    size_t max_num_blocks_per_seq;

    ptrdiff_t q_stride;
    ptrdiff_t k_batch_stride;
    ptrdiff_t k_row_stride;
    ptrdiff_t k_head_stride;
    ptrdiff_t v_batch_stride;
    ptrdiff_t v_row_stride;
    ptrdiff_t v_head_stride;
    ptrdiff_t o_stride;

    ptrdiff_t block_table_batch_stride;
    ptrdiff_t cache_lens_stride;

    static utils::Result<PagedAttentionV1Info> create(
        infiniopTensorDescriptor_t out_desc,          // [num_seqs, num_heads, head_size]
        infiniopTensorDescriptor_t query_desc,        // [num_seqs, num_heads, head_size]
        infiniopTensorDescriptor_t key_cache_desc,    // [num_blocks, num_heads, head_size/x, block_size, x]
        infiniopTensorDescriptor_t value_cache_desc,  // [num_blocks, num_heads, head_size, block_size]
        infiniopTensorDescriptor_t block_tables_desc, // [num_seqs, max_num_blocks_per_seq]
        infiniopTensorDescriptor_t seq_lens_desc,     // [num_seqs]
        const std::optional<infiniopTensorDescriptor_t> &alibi_slopes_desc,
        double scale) {

        auto dtype = query_desc->dtype();
        CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32);
        if (out_desc->dtype() != dtype || key_cache_desc->dtype() != dtype || value_cache_desc->dtype() != dtype) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        if (query_desc->ndim() != 3 || out_desc->ndim() != 3) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        // key_cache can be 4D [num_blocks, num_kv_heads, block_size, head_size] or
        // 5D [num_blocks, num_kv_heads, head_size/x, block_size, x]
        if ((key_cache_desc->ndim() != 4 && key_cache_desc->ndim() != 5) || (value_cache_desc->ndim() != 4 && value_cache_desc->ndim() != 5)) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        if (block_tables_desc->ndim() != 2 || seq_lens_desc->ndim() != 1) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        CHECK_OR_RETURN(query_desc->stride(2) == 1, INFINI_STATUS_BAD_TENSOR_STRIDES);
        CHECK_OR_RETURN(out_desc->stride(2) == 1, INFINI_STATUS_BAD_TENSOR_STRIDES);
        // For 4D cache: innermost dim is stride(3); for 5D cache: innermost dim is stride(4)
        CHECK_OR_RETURN(key_cache_desc->stride(key_cache_desc->ndim() - 1) == 1, INFINI_STATUS_BAD_TENSOR_STRIDES);
        CHECK_OR_RETURN(value_cache_desc->stride(value_cache_desc->ndim() - 1) == 1, INFINI_STATUS_BAD_TENSOR_STRIDES);

        const auto block_tables_dt = block_tables_desc->dtype();
        const auto seq_lens_dt = seq_lens_desc->dtype();
        const bool debug_dtype = (std::getenv("INFINIOP_FLASH_DEBUG_DTYPE") != nullptr);
        const bool block_tables_ok = (block_tables_dt == INFINI_DTYPE_I64); // int64
        const bool cache_lens_ok = (seq_lens_dt == INFINI_DTYPE_I64);       // int64
        if (!(block_tables_ok && cache_lens_ok)) {
            if (debug_dtype) {
                std::fprintf(stderr,
                             "[flash_attention] Bad index dtype: block_tables=%d cache_lens=%d (expected I32/I64/U32)\n",
                             static_cast<int>(block_tables_dt), static_cast<int>(seq_lens_dt));
            }
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        if (block_tables_dt != seq_lens_dt) {
            // Keep them consistent to simplify backend dispatch.
            if (debug_dtype) {
                std::fprintf(stderr,
                             "[flash_attention] Mismatched index dtype: block_tables=%d cache_lens=%d\n",
                             static_cast<int>(block_tables_dt), static_cast<int>(seq_lens_dt));
            }
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        CHECK_OR_RETURN(block_tables_desc->stride(1) == 1, INFINI_STATUS_BAD_TENSOR_STRIDES);
        CHECK_OR_RETURN(seq_lens_desc->stride(0) == 1, INFINI_STATUS_BAD_TENSOR_STRIDES);

        if (alibi_slopes_desc.has_value() && alibi_slopes_desc.value() != nullptr) {
            if (alibi_slopes_desc.value()->dtype() != INFINI_DTYPE_F32) {
                return INFINI_STATUS_BAD_TENSOR_DTYPE;
            }
            if (alibi_slopes_desc.value()->ndim() != 1) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
            CHECK_OR_RETURN(alibi_slopes_desc.value()->stride(0) == 1, INFINI_STATUS_BAD_TENSOR_STRIDES);
        }

        // Shapes
        auto query_shape = query_desc->shape();
        auto key_shape = key_cache_desc->shape();
        auto value_shape = value_cache_desc->shape();

        const size_t num_seqs = query_shape[0];
        const size_t num_heads = query_shape[1];
        const size_t head_size = query_shape[2];

        const size_t num_blocks = key_shape[0];
        const size_t num_kv_heads = key_shape[1];

        // For 4D: [num_blocks, num_kv_heads, block_size, head_size], block_size is at index 2
        // For 5D: [num_blocks, num_kv_heads, head_size/x, block_size, x], block_size is at index 3
        size_t page_block_size;
        if (key_cache_desc->ndim() == 4) {
            page_block_size = key_shape[2];
        } else {
            page_block_size = key_shape[3];
        }

        if (head_size != 64 && head_size != 128) {
            // First build only targets common FA2 head dims (expand later).
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        if (num_heads % num_kv_heads != 0) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        if (value_cache_desc->shape()[0] != key_shape[0] || value_cache_desc->shape()[1] != key_shape[1]) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        if (out_desc->shape()[0] != query_shape[0] || out_desc->shape()[1] != query_shape[1] || out_desc->shape()[2] != query_shape[2]) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        if (seq_lens_desc->shape()[0] != num_seqs) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        const size_t max_num_blocks_per_seq = block_tables_desc->shape()[1];

        // Strides (in elements)
        const ptrdiff_t q_stride = query_desc->stride(0);
        const ptrdiff_t o_stride = out_desc->stride(0);

        const ptrdiff_t k_batch_stride = key_cache_desc->stride(0);
        const ptrdiff_t k_head_stride = key_cache_desc->stride(1);
        // For 4D: row_stride is stride(2) (block_size dimension)
        // For 5D: row_stride is stride(3) (block_size dimension)
        const ptrdiff_t k_row_stride = key_cache_desc->ndim() == 4
                                         ? key_cache_desc->stride(2)
                                         : key_cache_desc->stride(3);

        const ptrdiff_t v_batch_stride = value_cache_desc->stride(0);
        const ptrdiff_t v_row_stride = value_cache_desc->ndim() == 4
                                         ? value_cache_desc->stride(2)
                                         : value_cache_desc->stride(3);
        const ptrdiff_t v_head_stride = value_cache_desc->stride(1);

        const ptrdiff_t block_table_batch_stride = block_tables_desc->stride(0);
        const ptrdiff_t cache_lens_stride = seq_lens_desc->stride(0);

        return utils::Result<PagedAttentionV1Info>(PagedAttentionV1Info{
            dtype,
            block_tables_dt,
            scale,
            num_seqs,
            num_heads,
            num_kv_heads,
            head_size,
            page_block_size,
            max_num_blocks_per_seq,
            q_stride,
            k_batch_stride,
            k_row_stride,
            k_head_stride,
            v_batch_stride,
            v_row_stride,
            v_head_stride,
            o_stride,
            block_table_batch_stride,
            cache_lens_stride,
        });
    }
};

} // namespace op::paged_attention_v1

#endif // __PAGED_ATTENTION_V1_INFO_H__
