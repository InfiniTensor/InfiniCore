// infiniop/ops/lightning_attention/info.h

#ifndef __LIGHTNING_ATTENTION_INFO_H__
#define __LIGHTNING_ATTENTION_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <vector>

namespace op {
namespace lightning_attention {

class LightningAttentionInfo {
    LightningAttentionInfo() = default;

public:
    infiniDtype_t data_dtype;
    infiniDtype_t index_dtype;
    size_t B, T, H, D, pool_size;

    std::vector<ptrdiff_t> out_strides;
    std::vector<ptrdiff_t> initial_state_strides;
    std::vector<ptrdiff_t> q_strides;
    std::vector<ptrdiff_t> k_strides;
    std::vector<ptrdiff_t> v_strides;
    std::vector<ptrdiff_t> slope_strides;
    std::vector<ptrdiff_t> initial_state_indices_strides;
    std::vector<ptrdiff_t> final_state_indices_strides;

    static utils::Result<LightningAttentionInfo>
    create(infiniopTensorDescriptor_t out_desc,
           infiniopTensorDescriptor_t initial_state_desc,
           infiniopTensorDescriptor_t q_desc,
           infiniopTensorDescriptor_t k_desc,
           infiniopTensorDescriptor_t v_desc,
           infiniopTensorDescriptor_t slope_desc,
           infiniopTensorDescriptor_t initial_state_indices_desc,
           infiniopTensorDescriptor_t final_state_indices_desc) {
        if (out_desc == nullptr || initial_state_desc == nullptr || q_desc == nullptr ||
            k_desc == nullptr || v_desc == nullptr || slope_desc == nullptr ||
            initial_state_indices_desc == nullptr || final_state_indices_desc == nullptr) {
            return INFINI_STATUS_NULL_POINTER;
        }

        auto data_dtype = q_desc->dtype();
        CHECK_DTYPE(data_dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32);
        if (k_desc->dtype() != data_dtype || v_desc->dtype() != data_dtype ||
            out_desc->dtype() != data_dtype || initial_state_desc->dtype() != data_dtype) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        if (slope_desc->dtype() != INFINI_DTYPE_F32) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        auto index_dtype = initial_state_indices_desc->dtype();
        CHECK_DTYPE(index_dtype, INFINI_DTYPE_I32, INFINI_DTYPE_I64);
        if (final_state_indices_desc->dtype() != index_dtype) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        const auto &q_shape = q_desc->shape();
        const auto &k_shape = k_desc->shape();
        const auto &v_shape = v_desc->shape();
        const auto &out_shape = out_desc->shape();
        const auto &state_shape = initial_state_desc->shape();
        const auto &slope_shape = slope_desc->shape();
        if (q_shape.size() != 4 || k_shape.size() != 4 || v_shape.size() != 4 || out_shape.size() != 4) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (k_shape != q_shape || v_shape != q_shape || out_shape != q_shape) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        const size_t B = q_shape[0];
        const size_t T = q_shape[1];
        const size_t H = q_shape[2];
        const size_t D = q_shape[3];
        if (B == 0 || T == 0 || H == 0 || D == 0) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (state_shape.size() != 4 || state_shape[1] != H || state_shape[2] != D || state_shape[3] != D) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        const size_t pool_size = state_shape[0];
        if (pool_size == 0) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (slope_shape.size() != 1 || slope_shape[0] != H) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (initial_state_indices_desc->shape().size() != 1 ||
            initial_state_indices_desc->shape()[0] != B ||
            final_state_indices_desc->shape().size() != 1 ||
            final_state_indices_desc->shape()[0] != B) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        // last dim must be contiguous for the sequence tensors (as in gated delta rule)
        if (q_desc->stride(3) != 1 || k_desc->stride(3) != 1 ||
            v_desc->stride(3) != 1 || out_desc->stride(3) != 1) {
            return INFINI_STATUS_BAD_TENSOR_STRIDES;
        }

        // Both implementations read the state indices with unit stride.
        if (initial_state_indices_desc->stride(0) != 1 ||
            final_state_indices_desc->stride(0) != 1) {
            return INFINI_STATUS_BAD_TENSOR_STRIDES;
        }

        LightningAttentionInfo info;
        info.data_dtype = data_dtype;
        info.index_dtype = index_dtype;
        info.B = B;
        info.T = T;
        info.H = H;
        info.D = D;
        info.pool_size = pool_size;
        info.out_strides = out_desc->strides();
        info.initial_state_strides = initial_state_desc->strides();
        info.q_strides = q_desc->strides();
        info.k_strides = k_desc->strides();
        info.v_strides = v_desc->strides();
        info.slope_strides = slope_desc->strides();
        info.initial_state_indices_strides = initial_state_indices_desc->strides();
        info.final_state_indices_strides = final_state_indices_desc->strides();
        return utils::Result<LightningAttentionInfo>(info);
    }
};

} // namespace lightning_attention
} // namespace op

#endif // __LIGHTNING_ATTENTION_INFO_H__


