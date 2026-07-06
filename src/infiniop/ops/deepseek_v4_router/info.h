#ifndef __DEEPSEEK_V4_ROUTER_INFO_H__
#define __DEEPSEEK_V4_ROUTER_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"

namespace op::deepseek_v4_router {

struct DeepseekV4TopkRouterInfo {
    size_t num_tokens;
    size_t num_experts;
    size_t topk;
    infiniDtype_t dtype;
    bool has_bias;
    bool renormalize;

    static utils::Result<DeepseekV4TopkRouterInfo> create(
        infiniopTensorDescriptor_t topk_weights_desc,
        infiniopTensorDescriptor_t topk_indices_desc,
        infiniopTensorDescriptor_t logits_desc,
        infiniopTensorDescriptor_t bias_desc,
        bool renormalize) {
        if (topk_weights_desc->dtype() != INFINI_DTYPE_F32 || topk_indices_desc->dtype() != INFINI_DTYPE_I32) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        auto dtype = logits_desc->dtype();
        if (dtype != INFINI_DTYPE_F16 && dtype != INFINI_DTYPE_BF16 && dtype != INFINI_DTYPE_F32) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        if (bias_desc != nullptr && bias_desc->dtype() != INFINI_DTYPE_F32) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        if (logits_desc->ndim() != 2 || topk_weights_desc->ndim() != 2 || topk_indices_desc->ndim() != 2) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        const size_t num_tokens = logits_desc->shape()[0];
        const size_t num_experts = logits_desc->shape()[1];
        const size_t topk = topk_weights_desc->shape()[1];
        if (topk == 0 || topk > num_experts || topk_weights_desc->shape()[0] != num_tokens || topk_indices_desc->shape()[0] != num_tokens || topk_indices_desc->shape()[1] != topk) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (bias_desc != nullptr && (bias_desc->ndim() != 1 || bias_desc->shape()[0] != num_experts)) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (logits_desc->strides()[1] != 1 || topk_weights_desc->strides()[1] != 1 || topk_indices_desc->strides()[1] != 1 || (bias_desc != nullptr && bias_desc->strides()[0] != 1)) {
            return INFINI_STATUS_BAD_TENSOR_STRIDES;
        }
        return utils::Result<DeepseekV4TopkRouterInfo>(DeepseekV4TopkRouterInfo{
            num_tokens,
            num_experts,
            topk,
            dtype,
            bias_desc != nullptr,
            renormalize,
        });
    }
};

struct DeepseekV4HashRouterInfo {
    size_t num_tokens;
    size_t num_experts;
    size_t topk;
    size_t vocab_size;
    infiniDtype_t logits_dtype;
    infiniDtype_t input_ids_dtype;
    infiniDtype_t tid2eid_dtype;
    bool renormalize;

    static utils::Result<DeepseekV4HashRouterInfo> create(
        infiniopTensorDescriptor_t topk_weights_desc,
        infiniopTensorDescriptor_t topk_indices_desc,
        infiniopTensorDescriptor_t logits_desc,
        infiniopTensorDescriptor_t input_ids_desc,
        infiniopTensorDescriptor_t tid2eid_desc,
        bool renormalize) {
        if (topk_weights_desc->dtype() != INFINI_DTYPE_F32 || topk_indices_desc->dtype() != INFINI_DTYPE_I32) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        auto logits_dtype = logits_desc->dtype();
        if (logits_dtype != INFINI_DTYPE_F16 && logits_dtype != INFINI_DTYPE_BF16 && logits_dtype != INFINI_DTYPE_F32) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        auto input_ids_dtype = input_ids_desc->dtype();
        auto tid2eid_dtype = tid2eid_desc->dtype();
        if ((input_ids_dtype != INFINI_DTYPE_I32 && input_ids_dtype != INFINI_DTYPE_I64)
            || (tid2eid_dtype != INFINI_DTYPE_I32 && tid2eid_dtype != INFINI_DTYPE_I64)) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        if (logits_desc->ndim() != 2 || topk_weights_desc->ndim() != 2 || topk_indices_desc->ndim() != 2 || tid2eid_desc->ndim() != 2) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (input_ids_desc->numel() != logits_desc->shape()[0]) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        const size_t num_tokens = logits_desc->shape()[0];
        const size_t num_experts = logits_desc->shape()[1];
        const size_t topk = topk_weights_desc->shape()[1];
        if (topk == 0 || topk > num_experts || topk_weights_desc->shape()[0] != num_tokens || topk_indices_desc->shape()[0] != num_tokens || topk_indices_desc->shape()[1] != topk) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (tid2eid_desc->shape()[1] != topk) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (logits_desc->strides()[1] != 1 || topk_weights_desc->strides()[1] != 1 || topk_indices_desc->strides()[1] != 1 || input_ids_desc->strides()[input_ids_desc->ndim() - 1] != 1 || tid2eid_desc->strides()[1] != 1) {
            return INFINI_STATUS_BAD_TENSOR_STRIDES;
        }
        return utils::Result<DeepseekV4HashRouterInfo>(DeepseekV4HashRouterInfo{
            num_tokens,
            num_experts,
            topk,
            tid2eid_desc->shape()[0],
            logits_dtype,
            input_ids_dtype,
            tid2eid_dtype,
            renormalize,
        });
    }
};

} // namespace op::deepseek_v4_router

#endif
