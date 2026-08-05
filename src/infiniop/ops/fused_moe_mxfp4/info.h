#ifndef __FUSED_MOE_MXFP4_INFO_H__
#define __FUSED_MOE_MXFP4_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include "infiniop/ops/fused_moe.h"

namespace op::fused_moe_mxfp4 {

class FusedMoeMxfp4Info {
    FusedMoeMxfp4Info() = default;

public:
    infiniDtype_t dtype;
    infiniopFusedMoeActivation_t activation;
    size_t num_tokens;
    size_t hidden_size;
    size_t intermediate_size;
    size_t num_experts;
    size_t topk;

    size_t routeCount() const { return num_tokens * topk; }

    static utils::Result<FusedMoeMxfp4Info> create(
        infiniopTensorDescriptor_t output_desc,
        infiniopTensorDescriptor_t input_desc,
        infiniopTensorDescriptor_t selected_experts_desc,
        infiniopTensorDescriptor_t routing_weights_desc,
        infiniopTensorDescriptor_t w13_packed_desc,
        infiniopTensorDescriptor_t w13_scale_desc,
        infiniopTensorDescriptor_t w2_packed_desc,
        infiniopTensorDescriptor_t w2_scale_desc,
        infiniopFusedMoeActivation_t activation) {
        CHECK_OR_RETURN(output_desc != nullptr && input_desc != nullptr
                            && selected_experts_desc != nullptr
                            && routing_weights_desc != nullptr
                            && w13_packed_desc != nullptr && w13_scale_desc != nullptr
                            && w2_packed_desc != nullptr && w2_scale_desc != nullptr,
                        INFINI_STATUS_NULL_POINTER);
        CHECK_OR_RETURN(activation == INFINIOP_FUSED_MOE_ACT_SWIGLU
                            || activation == INFINIOP_FUSED_MOE_ACT_SITUGLU,
                        INFINI_STATUS_BAD_PARAM);

        const auto dtype = input_desc->dtype();
        CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32);
        CHECK_OR_RETURN(output_desc->dtype() == dtype
                            && selected_experts_desc->dtype() == INFINI_DTYPE_I32
                            && routing_weights_desc->dtype() == INFINI_DTYPE_F32,
                        INFINI_STATUS_BAD_TENSOR_DTYPE);
        CHECK_DTYPE(w13_packed_desc->dtype(), INFINI_DTYPE_U8);
        CHECK_DTYPE(w13_scale_desc->dtype(), INFINI_DTYPE_U8);
        CHECK_DTYPE(w2_packed_desc->dtype(), INFINI_DTYPE_U8);
        CHECK_DTYPE(w2_scale_desc->dtype(), INFINI_DTYPE_U8);

        CHECK_OR_RETURN(input_desc->ndim() == 2 && output_desc->ndim() == 2
                            && selected_experts_desc->ndim() == 2
                            && routing_weights_desc->ndim() == 2
                            && w13_packed_desc->ndim() == 3
                            && w13_scale_desc->ndim() == 3
                            && w2_packed_desc->ndim() == 3
                            && w2_scale_desc->ndim() == 3,
                        INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(input_desc->isContiguous() && output_desc->isContiguous()
                            && selected_experts_desc->isContiguous()
                            && routing_weights_desc->isContiguous()
                            && w13_packed_desc->isContiguous()
                            && w13_scale_desc->isContiguous()
                            && w2_packed_desc->isContiguous()
                            && w2_scale_desc->isContiguous(),
                        INFINI_STATUS_BAD_TENSOR_STRIDES);

        const size_t T = input_desc->dim(0);
        const size_t H = input_desc->dim(1);
        const size_t E = w13_packed_desc->dim(0);
        const size_t two_I = w13_packed_desc->dim(1);
        const size_t topk = selected_experts_desc->dim(1);
        CHECK_OR_RETURN(T > 0 && H > 0 && H % 32 == 0 && E > 0
                            && two_I > 0 && two_I % 2 == 0 && topk > 0,
                        INFINI_STATUS_BAD_TENSOR_SHAPE);
        const size_t I = two_I / 2;
        CHECK_OR_RETURN(I % 32 == 0,
                        INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(output_desc->dim(0) == T && output_desc->dim(1) == H
                            && selected_experts_desc->dim(0) == T
                            && routing_weights_desc->dim(0) == T
                            && routing_weights_desc->dim(1) == topk
                            && w13_packed_desc->dim(2) == H / 2
                            && w13_scale_desc->dim(0) == E
                            && w13_scale_desc->dim(1) == two_I
                            && w13_scale_desc->dim(2) == H / 32
                            && w2_packed_desc->dim(0) == E
                            && w2_packed_desc->dim(1) == H
                            && w2_packed_desc->dim(2) == I / 2
                            && w2_scale_desc->dim(0) == E
                            && w2_scale_desc->dim(1) == H
                            && w2_scale_desc->dim(2) == I / 32,
                        INFINI_STATUS_BAD_TENSOR_SHAPE);

        FusedMoeMxfp4Info info;
        info.dtype = dtype;
        info.activation = activation;
        info.num_tokens = T;
        info.hidden_size = H;
        info.intermediate_size = I;
        info.num_experts = E;
        info.topk = topk;
        return utils::Result<FusedMoeMxfp4Info>(info);
    }
};

} // namespace op::fused_moe_mxfp4

#endif
