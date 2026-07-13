#ifndef __DEEPSEEK_V4_COMPRESSED_DECODE_INFO_H__
#define __DEEPSEEK_V4_COMPRESSED_DECODE_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"

#include <cstddef>
#include <cstdint>

namespace op::deepseek_v4_compressed_decode {

inline bool is_contiguous(infiniopTensorDescriptor_t desc) {
    ptrdiff_t expected = 1;
    for (ptrdiff_t i = static_cast<ptrdiff_t>(desc->ndim()) - 1; i >= 0; --i) {
        if (desc->stride(static_cast<size_t>(i)) != expected) {
            return false;
        }
        expected *= static_cast<ptrdiff_t>(desc->dim(static_cast<size_t>(i)));
    }
    return true;
}

inline bool is_activation_dtype(infiniDtype_t dtype) {
    return dtype == INFINI_DTYPE_F16 || dtype == INFINI_DTYPE_BF16 || dtype == INFINI_DTYPE_F32;
}

struct DeepseekV4CompressedDecodeInfo {
    size_t batch_size;
    size_t query_len;
    size_t num_heads;
    size_t key_len;
    size_t full_key_len;
    size_t key_offset;
    bool causal;
    size_t sliding_window;
    int64_t key_position_base;
    size_t num_kv_heads;
    size_t num_blocks;
    ptrdiff_t kv_comp_stride_batch;
    size_t head_dim;
    infiniDtype_t dtype;
    infiniDtype_t sink_dtype;
    infiniDtype_t positions_dtype;
    infiniDtype_t indexed_dtype;
    float softmax_scale;
    size_t compress_ratio;
    size_t index_top_k;
    size_t rope_dim;
    double rope_theta;
    bool use_yarn;
    double yarn_factor;
    double yarn_beta_fast;
    double yarn_beta_slow;
    int64_t yarn_original_seq_len;
    double yarn_extrapolation_factor;

    static utils::Result<DeepseekV4CompressedDecodeInfo> create(
        infiniopTensorDescriptor_t y_desc,
        infiniopTensorDescriptor_t q_desc,
        infiniopTensorDescriptor_t k_desc,
        infiniopTensorDescriptor_t kv_comp_desc,
        infiniopTensorDescriptor_t attn_sink_desc,
        infiniopTensorDescriptor_t query_positions_desc,
        infiniopTensorDescriptor_t block_positions_desc,
        infiniopTensorDescriptor_t indexed_blocks_desc,
        size_t key_offset,
        size_t active_key_len,
        bool causal,
        size_t sliding_window,
        int64_t key_position_base,
        float softmax_scale,
        size_t compress_ratio,
        size_t index_top_k,
        size_t rope_dim,
        double rope_theta,
        bool use_yarn,
        double yarn_factor,
        double yarn_beta_fast,
        double yarn_beta_slow,
        int64_t yarn_original_seq_len,
        double yarn_extrapolation_factor) {
        if (!y_desc || !q_desc || !k_desc || !kv_comp_desc || !attn_sink_desc ||
            !query_positions_desc || !block_positions_desc || !indexed_blocks_desc) {
            return INFINI_STATUS_NULL_POINTER;
        }
        const auto dtype = q_desc->dtype();
        if (!is_activation_dtype(dtype) || y_desc->dtype() != dtype ||
            k_desc->dtype() != dtype || kv_comp_desc->dtype() != dtype) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        const auto sink_dtype = attn_sink_desc->dtype();
        if (!is_activation_dtype(sink_dtype)) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        const auto positions_dtype = query_positions_desc->dtype();
        if ((positions_dtype != INFINI_DTYPE_I64 && positions_dtype != INFINI_DTYPE_I32) ||
            block_positions_desc->dtype() != positions_dtype) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        const auto indexed_dtype = indexed_blocks_desc->dtype();
        if (indexed_dtype != INFINI_DTYPE_I64 && indexed_dtype != INFINI_DTYPE_I32) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        if (y_desc->ndim() != 4 || q_desc->ndim() != 4 || k_desc->ndim() != 4 ||
            kv_comp_desc->ndim() != 3 || attn_sink_desc->ndim() != 1 ||
            query_positions_desc->ndim() != 1 || block_positions_desc->ndim() != 1 ||
            indexed_blocks_desc->ndim() != 1) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        const size_t batch_size = q_desc->dim(0);
        const size_t query_len = q_desc->dim(1);
        const size_t num_heads = q_desc->dim(2);
        const size_t head_dim = q_desc->dim(3);
        const size_t full_key_len = k_desc->dim(1);
        const size_t key_len = active_key_len == 0 ? full_key_len : active_key_len;
        const size_t num_kv_heads = k_desc->dim(2);
        const size_t num_blocks = kv_comp_desc->dim(1);
        if (batch_size == 0 || query_len == 0 || num_heads == 0 || full_key_len == 0 || key_len == 0 ||
            num_kv_heads == 0 || num_blocks == 0 || head_dim == 0 || compress_ratio == 0 || key_offset + key_len > full_key_len) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (num_heads % num_kv_heads != 0 || rope_dim > head_dim || (rope_dim % 2) != 0) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        const size_t compressed_keys = index_top_k == 0 ? num_blocks : index_top_k;
        if (compressed_keys + key_len > 4096) {
            return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
        }
        if (y_desc->dim(0) != batch_size || y_desc->dim(1) != query_len ||
            y_desc->dim(2) != num_heads || y_desc->dim(3) != head_dim ||
            k_desc->dim(0) != batch_size || k_desc->dim(3) != head_dim ||
            kv_comp_desc->dim(0) != batch_size || kv_comp_desc->dim(2) != head_dim ||
            attn_sink_desc->dim(0) != num_heads ||
            query_positions_desc->dim(0) != query_len ||
            block_positions_desc->dim(0) != num_blocks) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        const size_t expected_indexed = index_top_k == 0 ? 1 : batch_size * query_len * index_top_k;
        if (indexed_blocks_desc->dim(0) != expected_indexed) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (!is_contiguous(y_desc) || !is_contiguous(q_desc) || !is_contiguous(k_desc) ||
            !is_contiguous(attn_sink_desc) ||
            !is_contiguous(query_positions_desc) || !is_contiguous(block_positions_desc) ||
            !is_contiguous(indexed_blocks_desc)) {
            return INFINI_STATUS_BAD_TENSOR_STRIDES;
        }
        if (kv_comp_desc->stride(2) != 1
            || kv_comp_desc->stride(1) != static_cast<ptrdiff_t>(head_dim)
            || kv_comp_desc->stride(0)
                   < static_cast<ptrdiff_t>(num_blocks * head_dim)) {
            return INFINI_STATUS_BAD_TENSOR_STRIDES;
        }
        return utils::Result<DeepseekV4CompressedDecodeInfo>(DeepseekV4CompressedDecodeInfo{
            batch_size, query_len, num_heads, key_len, full_key_len, key_offset,
            causal, sliding_window, key_position_base,
            num_kv_heads, num_blocks, kv_comp_desc->stride(0), head_dim,
            dtype, sink_dtype, positions_dtype, indexed_dtype, softmax_scale, compress_ratio,
            index_top_k, rope_dim, rope_theta, use_yarn, yarn_factor, yarn_beta_fast,
            yarn_beta_slow, yarn_original_seq_len, yarn_extrapolation_factor});
    }
};

} // namespace op::deepseek_v4_compressed_decode

#endif
