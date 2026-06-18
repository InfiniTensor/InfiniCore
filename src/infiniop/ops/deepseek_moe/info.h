#ifndef __DEEPSEEK_MOE_INFO_H__
#define __DEEPSEEK_MOE_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <vector>

namespace op::deepseek_moe {

class DeepseekMoeInfo {
    DeepseekMoeInfo() = default;

public:
    infiniDtype_t dtype;
    size_t ntokens;
    size_t hidden_size;
    size_t topk;
    size_t intermediate_size;
    size_t num_experts;

    static utils::Result<DeepseekMoeInfo> create(
        infiniopTensorDescriptor_t out_desc,
        infiniopTensorDescriptor_t hidden_desc,
        infiniopTensorDescriptor_t topk_indices_desc,
        infiniopTensorDescriptor_t topk_weights_desc,
        size_t intermediate_size,
        size_t num_experts) {

        auto dtype = hidden_desc->dtype();
        if (dtype != INFINI_DTYPE_F16 && dtype != INFINI_DTYPE_BF16) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        if (out_desc->dtype() != dtype || topk_indices_desc->dtype() != INFINI_DTYPE_I32 || topk_weights_desc->dtype() != INFINI_DTYPE_F32) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        if (out_desc->ndim() != 2 || hidden_desc->ndim() != 2 || topk_indices_desc->ndim() != 2 || topk_weights_desc->ndim() != 2) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        auto hidden_shape = hidden_desc->shape();
        auto out_shape = out_desc->shape();
        auto indices_shape = topk_indices_desc->shape();
        auto weights_shape = topk_weights_desc->shape();
        if (out_shape != hidden_shape || indices_shape != weights_shape || indices_shape[0] != hidden_shape[0]) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (intermediate_size == 0 || num_experts == 0 || indices_shape[1] == 0 || indices_shape[1] > num_experts) {
            return INFINI_STATUS_BAD_PARAM;
        }
        if (hidden_desc->strides()[1] != 1 || out_desc->strides()[1] != 1 || topk_indices_desc->strides()[1] != 1 || topk_weights_desc->strides()[1] != 1) {
            return INFINI_STATUS_BAD_TENSOR_STRIDES;
        }

        return utils::Result<DeepseekMoeInfo>(DeepseekMoeInfo{
            dtype,
            hidden_shape[0],
            hidden_shape[1],
            indices_shape[1],
            intermediate_size,
            num_experts});
    }
};

} // namespace op::deepseek_moe

#endif
