#ifndef __MOE_FUSED_GATE_INFO_H__
#define __MOE_FUSED_GATE_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"

namespace op::moe_fused_gate {

struct MoeFusedGateInfo {
    size_t num_tokens;
    size_t num_experts;
    size_t topk;
    size_t num_expert_group;
    size_t topk_group;
    size_t num_fused_shared_experts;
    float routed_scaling_factor;
    bool apply_routed_scaling_factor_on_output;
    infiniDtype_t dtype;

    static utils::Result<MoeFusedGateInfo> create(
        infiniopTensorDescriptor_t topk_weights_desc,
        infiniopTensorDescriptor_t topk_indices_desc,
        infiniopTensorDescriptor_t input_desc,
        infiniopTensorDescriptor_t bias_desc,
        size_t num_expert_group,
        size_t topk_group,
        size_t num_fused_shared_experts,
        float routed_scaling_factor,
        bool apply_routed_scaling_factor_on_output) {
        if (topk_weights_desc->dtype() != INFINI_DTYPE_F32 || topk_indices_desc->dtype() != INFINI_DTYPE_I32 || bias_desc->dtype() != INFINI_DTYPE_F32) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        auto dtype = input_desc->dtype();
        if (dtype != INFINI_DTYPE_F16 && dtype != INFINI_DTYPE_BF16 && dtype != INFINI_DTYPE_F32) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        if (input_desc->ndim() != 2 || bias_desc->ndim() != 1 || topk_weights_desc->ndim() != 2 || topk_indices_desc->ndim() != 2) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        const size_t num_tokens = input_desc->shape()[0];
        const size_t num_experts = input_desc->shape()[1];
        const size_t topk = topk_weights_desc->shape()[1];
        if (num_expert_group == 0 || topk_group == 0 || num_experts % num_expert_group != 0 || num_experts / num_expert_group > 32 || topk <= num_fused_shared_experts || topk_weights_desc->shape()[0] != num_tokens || topk_indices_desc->shape()[0] != num_tokens || topk_indices_desc->shape()[1] != topk || bias_desc->shape()[0] != num_experts) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (input_desc->strides()[1] != 1 || bias_desc->strides()[0] != 1 || topk_weights_desc->strides()[1] != 1 || topk_indices_desc->strides()[1] != 1) {
            return INFINI_STATUS_BAD_TENSOR_STRIDES;
        }
        return utils::Result<MoeFusedGateInfo>(MoeFusedGateInfo{
            num_tokens,
            num_experts,
            topk,
            num_expert_group,
            topk_group,
            num_fused_shared_experts,
            routed_scaling_factor,
            apply_routed_scaling_factor_on_output,
            dtype,
        });
    }
};

} // namespace op::moe_fused_gate

#endif
