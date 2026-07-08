#ifndef DSV4_MASK_TOPK_IDS_INFO_H
#define DSV4_MASK_TOPK_IDS_INFO_H

#include "../../../utils.h"
#include "../../tensor.h"

namespace op::dsv4_mask_topk_ids {

struct Info {
    size_t batch;
    size_t topk;
};

inline infiniStatus_t createInfo(
    Info *info,
    infiniopTensorDescriptor_t topk_ids,
    infiniopTensorDescriptor_t num_token_non_padded) {
    CHECK_OR_RETURN(info && topk_ids && num_token_non_padded, INFINI_STATUS_NULL_POINTER);
    CHECK_OR_RETURN(topk_ids->dtype() == INFINI_DTYPE_I32, INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(num_token_non_padded->dtype() == INFINI_DTYPE_I32, INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(topk_ids->ndim() == 2, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(num_token_non_padded->numel() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(topk_ids->isContiguous() && num_token_non_padded->isContiguous(), INFINI_STATUS_BAD_TENSOR_STRIDES);
    *info = Info{topk_ids->dim(0), topk_ids->dim(1)};
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::dsv4_mask_topk_ids

#endif
