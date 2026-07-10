#ifndef __FUSED_MOE_INFO_H__
#define __FUSED_MOE_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include "infiniop/ops/fused_moe.h"

namespace op::fused_moe {

class FusedMoeInfo {
    FusedMoeInfo() = default;

public:
    infiniDtype_t dtype;
    bool has_b1;
    bool has_b2;
    infiniopFusedMoeActivation_t activation;
    size_t N;
    size_t hidden_size;
    size_t inter_size;
    size_t num_experts;
    size_t topk;
    size_t w1_cols;

    static utils::Result<FusedMoeInfo> create(
        infiniopTensorDescriptor_t out_desc,
        infiniopTensorDescriptor_t input_desc,
        infiniopTensorDescriptor_t token_selected_experts_desc,
        infiniopTensorDescriptor_t token_final_scales_desc,
        infiniopTensorDescriptor_t w1_desc,
        infiniopTensorDescriptor_t w2_desc,
        infiniopTensorDescriptor_t b1_desc,
        infiniopTensorDescriptor_t b2_desc,
        infiniopFusedMoeActivation_t activation) {

        if (out_desc == nullptr || input_desc == nullptr ||
            token_selected_experts_desc == nullptr || token_final_scales_desc == nullptr ||
            w1_desc == nullptr || w2_desc == nullptr) {
            return INFINI_STATUS_NULL_POINTER;
        }
        if (activation != INFINIOP_FUSED_MOE_ACT_SILU && activation != INFINIOP_FUSED_MOE_ACT_SWIGLU) {
            return INFINI_STATUS_BAD_PARAM;
        }

        auto dtype = input_desc->dtype();
        CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32);

        if (out_desc->dtype() != dtype || w1_desc->dtype() != dtype || w2_desc->dtype() != dtype ||
            (b1_desc != nullptr && b1_desc->dtype() != dtype) ||
            (b2_desc != nullptr && b2_desc->dtype() != dtype)) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        if (token_selected_experts_desc->dtype() != INFINI_DTYPE_I32 ||
            token_final_scales_desc->dtype() != INFINI_DTYPE_F32) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        if (input_desc->ndim() != 2 || out_desc->ndim() != 2 ||
            token_selected_experts_desc->ndim() != 2 || token_final_scales_desc->ndim() != 2 ||
            w1_desc->ndim() != 3 || w2_desc->ndim() != 3 ||
            (b1_desc != nullptr && b1_desc->ndim() != 2) ||
            (b2_desc != nullptr && b2_desc->ndim() != 2)) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        auto input_shape = input_desc->shape();
        auto out_shape = out_desc->shape();
        auto indices_shape = token_selected_experts_desc->shape();
        auto scales_shape = token_final_scales_desc->shape();
        auto w1_shape = w1_desc->shape();
        auto w2_shape = w2_desc->shape();

        size_t N = input_shape[0];
        size_t hidden = input_shape[1];
        size_t topk = indices_shape[1];
        size_t experts = w1_shape[0];
        size_t w1_cols = w1_shape[1];
        size_t inter = w2_shape[2];

        if (out_shape[0] != N || out_shape[1] != hidden ||
            indices_shape[0] != N || scales_shape[0] != N || scales_shape[1] != topk ||
            w1_shape[2] != hidden || w2_shape[0] != experts || w2_shape[1] != hidden) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (activation == INFINIOP_FUSED_MOE_ACT_SWIGLU) {
            if (w1_cols != 2 * inter) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
        } else if (w1_cols != inter) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        if (b1_desc != nullptr) {
            auto b1_shape = b1_desc->shape();
            if (b1_shape[0] != experts || b1_shape[1] != w1_cols) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
        }
        if (b2_desc != nullptr) {
            auto b2_shape = b2_desc->shape();
            if (b2_shape[0] != experts || b2_shape[1] != hidden) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
        }

        FusedMoeInfo info;
        info.dtype = dtype;
        info.has_b1 = b1_desc != nullptr;
        info.has_b2 = b2_desc != nullptr;
        info.activation = activation;
        info.N = N;
        info.hidden_size = hidden;
        info.inter_size = inter;
        info.num_experts = experts;
        info.topk = topk;
        info.w1_cols = w1_cols;
        return utils::Result<FusedMoeInfo>(info);
    }
};

} // namespace op::fused_moe

#endif // __FUSED_MOE_INFO_H__
