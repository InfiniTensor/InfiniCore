#ifndef __FLASH_ATTENTION_INFO_H__
#define __FLASH_ATTENTION_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <cstdint>
#include <vector>

namespace op::flash_attention {

class FlashAttentionInfo {
    FlashAttentionInfo() = default;

public:
    infiniDtype_t dtype;
    float scale;
    char is_causal;

    // Dimensions
    size_t batch_size;
    size_t num_heads;
    size_t num_kv_heads;
    size_t seq_len_q;
    size_t seq_len_kv;
    size_t head_dim;

    // Strides for Q/K/V/Out (batch, head, seq, dim)
    ptrdiff_t q_stride_batch;
    ptrdiff_t q_stride_head;
    ptrdiff_t q_stride_seq;
    ptrdiff_t q_stride_dim;

    ptrdiff_t k_stride_batch;
    ptrdiff_t k_stride_head;
    ptrdiff_t k_stride_seq;
    ptrdiff_t k_stride_dim;

    ptrdiff_t v_stride_batch;
    ptrdiff_t v_stride_head;
    ptrdiff_t v_stride_seq;
    ptrdiff_t v_stride_dim;

    ptrdiff_t out_stride_batch;
    ptrdiff_t out_stride_head;
    ptrdiff_t out_stride_seq;
    ptrdiff_t out_stride_dim;

    // Total KV length (for variable sequence lengths)
    bool has_variable_kv_len;

    static utils::Result<FlashAttentionInfo> create(
        infiniopTensorDescriptor_t out_desc,
        infiniopTensorDescriptor_t q_desc,
        infiniopTensorDescriptor_t k_desc,
        infiniopTensorDescriptor_t v_desc,
        infiniopTensorDescriptor_t total_kv_len_desc,
        float scale,
        char is_causal) {

        // Check dtypes
        auto dtype = q_desc->dtype();
        if (out_desc->dtype() != dtype || k_desc->dtype() != dtype || v_desc->dtype() != dtype) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32);

        // Check ndim
        if (q_desc->ndim() != 4 || k_desc->ndim() != 4 || v_desc->ndim() != 4 || out_desc->ndim() != 4) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        // Get dimensions
        size_t batch_size = q_desc->dim(0);
        size_t num_heads = q_desc->dim(1);
        size_t seq_len_q = q_desc->dim(2);
        size_t head_dim = q_desc->dim(3);

        size_t num_kv_heads = k_desc->dim(1);
        size_t seq_len_kv = k_desc->dim(2);
        size_t kv_head_dim = k_desc->dim(3);

        // Validate shapes
        if (out_desc->dim(0) != batch_size || out_desc->dim(1) != num_heads || out_desc->dim(2) != seq_len_q || out_desc->dim(3) != head_dim) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        if (k_desc->dim(0) != batch_size || k_desc->dim(3) != head_dim || v_desc->dim(0) != batch_size || v_desc->dim(1) != num_kv_heads || v_desc->dim(2) != seq_len_kv || v_desc->dim(3) != head_dim) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        // Check total_kv_len_desc
        bool has_variable_kv_len = (total_kv_len_desc != nullptr);
        if (has_variable_kv_len) {
            if (total_kv_len_desc->ndim() != 1 || total_kv_len_desc->dim(0) != batch_size || total_kv_len_desc->dtype() != INFINI_DTYPE_I64) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
        }

        // Check contiguity of last dimension
        if (q_desc->stride(3) != 1 || k_desc->stride(3) != 1 || v_desc->stride(3) != 1 || out_desc->stride(3) != 1) {
            return INFINI_STATUS_BAD_TENSOR_STRIDES;
        }

        FlashAttentionInfo info;
        info.dtype = dtype;
        info.scale = scale;
        info.is_causal = is_causal;
        info.batch_size = batch_size;
        info.num_heads = num_heads;
        info.num_kv_heads = num_kv_heads;
        info.seq_len_q = seq_len_q;
        info.seq_len_kv = seq_len_kv;
        info.head_dim = head_dim;
        info.has_variable_kv_len = has_variable_kv_len;

        // Strides
        info.q_stride_batch = q_desc->stride(0);
        info.q_stride_head = q_desc->stride(1);
        info.q_stride_seq = q_desc->stride(2);
        info.q_stride_dim = q_desc->stride(3);

        info.k_stride_batch = k_desc->stride(0);
        info.k_stride_head = k_desc->stride(1);
        info.k_stride_seq = k_desc->stride(2);
        info.k_stride_dim = k_desc->stride(3);

        info.v_stride_batch = v_desc->stride(0);
        info.v_stride_head = v_desc->stride(1);
        info.v_stride_seq = v_desc->stride(2);
        info.v_stride_dim = v_desc->stride(3);

        info.out_stride_batch = out_desc->stride(0);
        info.out_stride_head = out_desc->stride(1);
        info.out_stride_seq = out_desc->stride(2);
        info.out_stride_dim = out_desc->stride(3);

        return utils::Result<FlashAttentionInfo>(info);
    }
};

} // namespace op::flash_attention

#endif // __FLASH_ATTENTION_INFO_H__
