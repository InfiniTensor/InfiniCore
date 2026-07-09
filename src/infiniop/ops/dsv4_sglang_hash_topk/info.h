#ifndef DSV4_SGLANG_HASH_TOPK_INFO_H
#define DSV4_SGLANG_HASH_TOPK_INFO_H

#include "../../../utils.h"
#include "../../tensor.h"

namespace op::dsv4_sglang_hash_topk {

struct Info {
    size_t num_tokens;
    size_t num_experts;
    size_t topk;
    size_t output_width;
    float routed_scaling_factor;
};

inline infiniStatus_t createInfo(Info *info,
                                 infiniopTensorDescriptor_t router_logits_desc,
                                 infiniopTensorDescriptor_t input_ids_desc,
                                 infiniopTensorDescriptor_t tid2eid_desc,
                                 infiniopTensorDescriptor_t topk_weights_desc,
                                 infiniopTensorDescriptor_t topk_ids_desc,
                                 float routed_scaling_factor) {
    CHECK_OR_RETURN(info != nullptr && router_logits_desc != nullptr && input_ids_desc != nullptr && tid2eid_desc != nullptr && topk_weights_desc != nullptr && topk_ids_desc != nullptr, INFINI_STATUS_NULL_POINTER);
    CHECK_OR_RETURN(router_logits_desc->dtype() == INFINI_DTYPE_F32 && input_ids_desc->dtype() == INFINI_DTYPE_I64 && tid2eid_desc->dtype() == INFINI_DTYPE_I32 && topk_weights_desc->dtype() == INFINI_DTYPE_F32 && topk_ids_desc->dtype() == INFINI_DTYPE_I32,
                    INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(router_logits_desc->ndim() == 2 && input_ids_desc->ndim() == 1 && tid2eid_desc->ndim() == 2 && topk_weights_desc->ndim() == 2 && topk_ids_desc->ndim() == 2,
                    INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(router_logits_desc->isContiguous() && input_ids_desc->isContiguous() && tid2eid_desc->isContiguous() && topk_weights_desc->isContiguous() && topk_ids_desc->isContiguous(),
                    INFINI_STATUS_BAD_TENSOR_STRIDES);

    const auto num_tokens = router_logits_desc->dim(0);
    const auto num_experts = router_logits_desc->dim(1);
    const auto topk = tid2eid_desc->dim(1);
    const auto output_width = topk_weights_desc->dim(1);
    CHECK_OR_RETURN(num_tokens > 0 && num_experts > 0 && topk > 0 && output_width > topk, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(input_ids_desc->dim(0) == num_tokens && tid2eid_desc->dim(0) == num_tokens && topk_weights_desc->dim(0) == num_tokens && topk_ids_desc->dim(0) == num_tokens && topk_ids_desc->dim(1) == output_width,
                    INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(topk <= num_experts, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(routed_scaling_factor != 0.0f, INFINI_STATUS_BAD_PARAM);

    *info = Info{num_tokens, num_experts, topk, output_width, routed_scaling_factor};
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::dsv4_sglang_hash_topk

#endif
