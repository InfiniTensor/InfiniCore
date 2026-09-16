#ifndef __FUSED_MOE_W4A8_INFO_H__
#define __FUSED_MOE_W4A8_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include "infiniop/ops/fused_moe.h"

namespace op::fused_moe_w4a8 {

inline size_t alignWorkspace(size_t size) {
    constexpr size_t alignment = 256;
    return (size + alignment - 1) / alignment * alignment;
}

class FusedMoeW4A8Info {
    FusedMoeW4A8Info() = default;

public:
    infiniDtype_t dtype;
    infiniopFusedMoeActivation_t activation;
    size_t num_tokens;
    size_t hidden_size;
    size_t intermediate_size;
    size_t num_experts;
    size_t topk;
    bool weights_are_aiter_shuffled;

    size_t routeCount() const { return num_tokens * topk; }
    size_t dtypeSize() const { return dtype == INFINI_DTYPE_F32 ? 4 : 2; }
    size_t quantizedInputBytes() const {
        const size_t rows = weights_are_aiter_shuffled ? routeCount() : num_tokens;
        return alignWorkspace(rows * hidden_size);
    }
    size_t inputScaleBytes() const {
        const size_t rows = weights_are_aiter_shuffled ? routeCount() : num_tokens;
        return alignWorkspace(rows * sizeof(float));
    }
    size_t activatedBytes() const {
        return alignWorkspace(routeCount() * intermediate_size * dtypeSize());
    }
    size_t quantizedActivatedBytes() const {
        return alignWorkspace(routeCount() * intermediate_size);
    }
    size_t activatedScaleBytes() const {
        return alignWorkspace(routeCount() * sizeof(float));
    }
    size_t aiterBlockSizeM() const {
        if (num_tokens <= 128) {
            return 16;
        }
        if (num_tokens <= 512) {
            return 32;
        }
        if (num_tokens <= 1024) {
            return 64;
        }
        if (num_tokens <= 2048) {
            return 48;
        }
        return 64;
    }
    size_t aiterSortedTokenCapacity() const {
        return routeCount() + num_experts * (aiterBlockSizeM() - 1);
    }
    size_t aiterGemm1Bytes() const {
        return alignWorkspace(routeCount() * 2 * intermediate_size * dtypeSize());
    }
    size_t aiterGemm2Bytes() const {
        return alignWorkspace(routeCount() * hidden_size * dtypeSize());
    }
    size_t aiterSortedTokenBytes() const {
        return alignWorkspace(aiterSortedTokenCapacity() * sizeof(int32_t));
    }
    size_t aiterExpertBlockBytes() const {
        return alignWorkspace(
            (aiterSortedTokenCapacity() + aiterBlockSizeM() - 1)
            / aiterBlockSizeM() * sizeof(int32_t));
    }
    size_t aiterRoutingMetadataBytes() const {
        return alignWorkspace((3 * num_experts + 2) * sizeof(int32_t));
    }
    size_t aiterRoutingBytes() const {
        return aiterSortedTokenBytes() + aiterExpertBlockBytes()
             + aiterRoutingMetadataBytes();
    }
    size_t workspaceSize() const {
        const size_t common = quantizedInputBytes() + inputScaleBytes()
                            + activatedBytes() + quantizedActivatedBytes()
                            + activatedScaleBytes();
        if (!weights_are_aiter_shuffled) {
            return common;
        }
        return common + aiterGemm1Bytes() + aiterGemm2Bytes()
             + aiterRoutingBytes();
    }

    static utils::Result<FusedMoeW4A8Info> create(
        infiniopTensorDescriptor_t output_desc,
        infiniopTensorDescriptor_t input_desc,
        infiniopTensorDescriptor_t selected_experts_desc,
        infiniopTensorDescriptor_t routing_weights_desc,
        infiniopTensorDescriptor_t w13_packed_desc,
        infiniopTensorDescriptor_t w13_scale_desc,
        infiniopTensorDescriptor_t w2_packed_desc,
        infiniopTensorDescriptor_t w2_scale_desc,
        infiniopFusedMoeActivation_t activation,
        bool weights_are_aiter_shuffled) {
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
                            && routing_weights_desc->dtype() == INFINI_DTYPE_F32
                            && w13_packed_desc->dtype() == INFINI_DTYPE_I8
                            && w13_scale_desc->dtype() == INFINI_DTYPE_F32
                            && w2_packed_desc->dtype() == INFINI_DTYPE_I8
                            && w2_scale_desc->dtype() == INFINI_DTYPE_F32,
                        INFINI_STATUS_BAD_TENSOR_DTYPE);
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
        CHECK_OR_RETURN(T > 0 && H > 0 && H % 2 == 0 && E > 0
                            && two_I > 0 && two_I % 2 == 0 && topk > 0,
                        INFINI_STATUS_BAD_TENSOR_SHAPE);
        const size_t I = two_I / 2;
        CHECK_OR_RETURN(I % 2 == 0
                            && output_desc->dim(0) == T && output_desc->dim(1) == H
                            && selected_experts_desc->dim(0) == T
                            && routing_weights_desc->dim(0) == T
                            && routing_weights_desc->dim(1) == topk
                            && w13_packed_desc->dim(2) == H / 2
                            && w13_scale_desc->dim(0) == E
                            && w13_scale_desc->dim(1) == two_I
                            && w13_scale_desc->dim(2) == 1
                            && w2_packed_desc->dim(0) == E
                            && w2_packed_desc->dim(1) == H
                            && w2_packed_desc->dim(2) == I / 2
                            && w2_scale_desc->dim(0) == E
                            && w2_scale_desc->dim(1) == H
                            && w2_scale_desc->dim(2) == 1,
                        INFINI_STATUS_BAD_TENSOR_SHAPE);

        return utils::Result<FusedMoeW4A8Info>(
            FusedMoeW4A8Info{dtype, activation, T, H, I, E, topk,
                             weights_are_aiter_shuffled});
    }
};

} // namespace op::fused_moe_w4a8

#endif
