#ifndef __MOE_TOPK_SIGMOID_INFO_H__
#define __MOE_TOPK_SIGMOID_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"

namespace op::moe_topk_sigmoid {

struct MoeTopkSigmoidInfo {
    size_t num_tokens;
    size_t num_experts;
    size_t topk;
    infiniDtype_t dtype;
    bool has_correction_bias;
    bool renormalize;

    static utils::Result<MoeTopkSigmoidInfo> create(
        infiniopTensorDescriptor_t topk_weights_desc,
        infiniopTensorDescriptor_t topk_indices_desc,
        infiniopTensorDescriptor_t gating_output_desc,
        infiniopTensorDescriptor_t correction_bias_desc,
        bool renormalize) {
        if (topk_weights_desc->dtype() != INFINI_DTYPE_F32 || topk_indices_desc->dtype() != INFINI_DTYPE_I32) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        auto dtype = gating_output_desc->dtype();
        if (dtype != INFINI_DTYPE_F16 && dtype != INFINI_DTYPE_BF16 && dtype != INFINI_DTYPE_F32) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        if (correction_bias_desc != nullptr && correction_bias_desc->dtype() != INFINI_DTYPE_F32) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        if (gating_output_desc->ndim() != 2 || topk_weights_desc->ndim() != 2 || topk_indices_desc->ndim() != 2) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        const size_t num_tokens = gating_output_desc->shape()[0];
        const size_t num_experts = gating_output_desc->shape()[1];
        const size_t topk = topk_weights_desc->shape()[1];
        if (topk == 0 || topk > num_experts || topk_weights_desc->shape()[0] != num_tokens || topk_indices_desc->shape()[0] != num_tokens || topk_indices_desc->shape()[1] != topk) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (correction_bias_desc != nullptr && (correction_bias_desc->ndim() != 1 || correction_bias_desc->shape()[0] != num_experts)) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (gating_output_desc->strides()[1] != 1 || topk_weights_desc->strides()[1] != 1 || topk_indices_desc->strides()[1] != 1 || (correction_bias_desc != nullptr && correction_bias_desc->strides()[0] != 1)) {
            return INFINI_STATUS_BAD_TENSOR_STRIDES;
        }
        return utils::Result<MoeTopkSigmoidInfo>(MoeTopkSigmoidInfo{
            num_tokens,
            num_experts,
            topk,
            dtype,
            correction_bias_desc != nullptr,
            renormalize,
        });
    }
};

} // namespace op::moe_topk_sigmoid

#endif
