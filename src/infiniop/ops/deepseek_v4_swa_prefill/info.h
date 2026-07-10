#ifndef __DEEPSEEK_V4_SWA_PREFILL_INFO_H__
#define __DEEPSEEK_V4_SWA_PREFILL_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"

#include <cstddef>
#include <cstdint>

namespace op::deepseek_v4_swa_prefill {

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

struct DeepseekV4SwaPrefillInfo {
    size_t batch_size;
    size_t query_len;
    size_t num_heads;
    size_t key_len;
    size_t num_kv_heads;
    size_t head_dim;
    infiniDtype_t dtype;
    infiniDtype_t sink_dtype;
    infiniDtype_t query_positions_dtype;
    infiniDtype_t key_positions_dtype;
    float softmax_scale;
    size_t window;
    size_t rope_dim;
    double rope_theta;
    bool use_yarn;
    double yarn_factor;
    double yarn_beta_fast;
    double yarn_beta_slow;
    int64_t yarn_original_seq_len;
    double yarn_extrapolation_factor;

    static utils::Result<DeepseekV4SwaPrefillInfo> create(
        infiniopTensorDescriptor_t y_desc,
        infiniopTensorDescriptor_t q_desc,
        infiniopTensorDescriptor_t k_desc,
        infiniopTensorDescriptor_t attn_sink_desc,
        infiniopTensorDescriptor_t query_positions_desc,
        infiniopTensorDescriptor_t key_positions_desc,
        float softmax_scale,
        size_t window,
        size_t rope_dim,
        double rope_theta,
        bool use_yarn,
        double yarn_factor,
        double yarn_beta_fast,
        double yarn_beta_slow,
        int64_t yarn_original_seq_len,
        double yarn_extrapolation_factor) {
        if (!y_desc || !q_desc || !k_desc || !attn_sink_desc || !query_positions_desc || !key_positions_desc) {
            return INFINI_STATUS_NULL_POINTER;
        }
        const auto dtype = q_desc->dtype();
        if (!is_activation_dtype(dtype) || y_desc->dtype() != dtype || k_desc->dtype() != dtype) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        const auto sink_dtype = attn_sink_desc->dtype();
        if (!is_activation_dtype(sink_dtype)) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        const auto qpos_dtype = query_positions_desc->dtype();
        const auto kpos_dtype = key_positions_desc->dtype();
        if ((qpos_dtype != INFINI_DTYPE_I64 && qpos_dtype != INFINI_DTYPE_I32)
            || (kpos_dtype != INFINI_DTYPE_I64 && kpos_dtype != INFINI_DTYPE_I32)) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        if (y_desc->ndim() != 4 || q_desc->ndim() != 4 || k_desc->ndim() != 4 ||
            attn_sink_desc->ndim() != 1 || query_positions_desc->ndim() != 1 || key_positions_desc->ndim() != 1) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        const size_t batch_size = q_desc->dim(0);
        const size_t query_len = q_desc->dim(1);
        const size_t num_heads = q_desc->dim(2);
        const size_t head_dim = q_desc->dim(3);
        const size_t key_len = k_desc->dim(1);
        const size_t num_kv_heads = k_desc->dim(2);
        if (batch_size == 0 || query_len == 0 || num_heads == 0 || key_len == 0 || num_kv_heads == 0 || head_dim == 0) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (num_heads % num_kv_heads != 0 || rope_dim > head_dim || (rope_dim % 2) != 0) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (y_desc->dim(0) != batch_size || y_desc->dim(1) != query_len || y_desc->dim(2) != num_heads || y_desc->dim(3) != head_dim ||
            k_desc->dim(0) != batch_size || k_desc->dim(3) != head_dim || attn_sink_desc->dim(0) != num_heads ||
            query_positions_desc->dim(0) != query_len || key_positions_desc->dim(0) != key_len) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (!is_contiguous(y_desc) || !is_contiguous(q_desc) || !is_contiguous(k_desc) ||
            !is_contiguous(attn_sink_desc) || !is_contiguous(query_positions_desc) || !is_contiguous(key_positions_desc)) {
            return INFINI_STATUS_BAD_TENSOR_STRIDES;
        }
        return utils::Result<DeepseekV4SwaPrefillInfo>(DeepseekV4SwaPrefillInfo{
            batch_size, query_len, num_heads, key_len, num_kv_heads, head_dim,
            dtype, sink_dtype, qpos_dtype, kpos_dtype, softmax_scale, window, rope_dim, rope_theta,
            use_yarn, yarn_factor, yarn_beta_fast, yarn_beta_slow,
            yarn_original_seq_len, yarn_extrapolation_factor});
    }
};

} // namespace op::deepseek_v4_swa_prefill

#endif
