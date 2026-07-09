#ifndef DSV4_SGLANG_MASK_TOPK_IDS_INFO_H
#define DSV4_SGLANG_MASK_TOPK_IDS_INFO_H

#include "../../../utils.h"
#include "../../tensor.h"

namespace op::dsv4_sglang_mask_topk_ids {

struct Info {
    size_t batch;
    size_t topk;
};

inline infiniStatus_t createInfo(Info *info,
                                 infiniopTensorDescriptor_t topk_ids_desc,
                                 infiniopTensorDescriptor_t num_token_non_padded_desc) {
    CHECK_OR_RETURN(info != nullptr && topk_ids_desc != nullptr && num_token_non_padded_desc != nullptr, INFINI_STATUS_NULL_POINTER);
    CHECK_OR_RETURN(topk_ids_desc->dtype() == INFINI_DTYPE_I32 && num_token_non_padded_desc->dtype() == INFINI_DTYPE_I32, INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(topk_ids_desc->ndim() == 2, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN((num_token_non_padded_desc->ndim() == 0) || (num_token_non_padded_desc->ndim() == 1 && num_token_non_padded_desc->dim(0) == 1), INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(topk_ids_desc->isContiguous() && num_token_non_padded_desc->isContiguous(), INFINI_STATUS_BAD_TENSOR_STRIDES);
    CHECK_OR_RETURN(topk_ids_desc->dim(0) > 0 && topk_ids_desc->dim(1) > 0, INFINI_STATUS_BAD_TENSOR_SHAPE);

    *info = Info{topk_ids_desc->dim(0), topk_ids_desc->dim(1)};
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::dsv4_sglang_mask_topk_ids

#endif
