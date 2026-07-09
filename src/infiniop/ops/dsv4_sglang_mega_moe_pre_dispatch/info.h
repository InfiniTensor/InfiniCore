#ifndef DSV4_SGLANG_MEGA_MOE_PRE_DISPATCH_INFO_H
#define DSV4_SGLANG_MEGA_MOE_PRE_DISPATCH_INFO_H
#include "../../../utils.h"
#include "../../tensor.h"
namespace op::dsv4_sglang_mega_moe_pre_dispatch {
struct Info {
    size_t tokens;
    size_t padded;
    size_t hidden;
    size_t topk;
    size_t sf_cols;
};
inline infiniStatus_t createInfo(Info *info, infiniopTensorDescriptor_t x_desc, infiniopTensorDescriptor_t topk_idx_desc, infiniopTensorDescriptor_t topk_weights_desc, infiniopTensorDescriptor_t buf_x_desc, infiniopTensorDescriptor_t buf_x_sf_desc, infiniopTensorDescriptor_t buf_topk_idx_desc, infiniopTensorDescriptor_t buf_topk_weights_desc) {
    CHECK_OR_RETURN(info && x_desc && topk_idx_desc && topk_weights_desc && buf_x_desc && buf_x_sf_desc && buf_topk_idx_desc && buf_topk_weights_desc, INFINI_STATUS_NULL_POINTER);
    CHECK_OR_RETURN(x_desc->dtype() == INFINI_DTYPE_BF16 && topk_idx_desc->dtype() == INFINI_DTYPE_I32 && topk_weights_desc->dtype() == INFINI_DTYPE_F32 && buf_x_desc->dtype() == INFINI_DTYPE_I8 && buf_x_sf_desc->dtype() == INFINI_DTYPE_I32 && buf_topk_idx_desc->dtype() == INFINI_DTYPE_I64 && buf_topk_weights_desc->dtype() == INFINI_DTYPE_F32, INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(x_desc->ndim() == 2 && topk_idx_desc->ndim() == 2 && topk_weights_desc->ndim() == 2 && buf_x_desc->ndim() == 2 && buf_x_sf_desc->ndim() == 2 && buf_topk_idx_desc->ndim() == 2 && buf_topk_weights_desc->ndim() == 2, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(x_desc->isContiguous() && topk_idx_desc->isContiguous() && topk_weights_desc->isContiguous() && buf_x_desc->isContiguous() && buf_x_sf_desc->isContiguous() && buf_topk_idx_desc->isContiguous() && buf_topk_weights_desc->isContiguous(), INFINI_STATUS_BAD_TENSOR_STRIDES);
    size_t tokens = x_desc->dim(0), hidden = x_desc->dim(1), topk = topk_idx_desc->dim(1), padded = buf_x_desc->dim(0);
    CHECK_OR_RETURN(hidden % 32 == 0 && topk_idx_desc->dim(0) == tokens && topk_weights_desc->dim(0) == tokens && topk_weights_desc->dim(1) == topk, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(buf_x_desc->dim(1) == hidden && buf_topk_idx_desc->dim(0) == padded && buf_topk_idx_desc->dim(1) == topk && buf_topk_weights_desc->dim(0) == padded && buf_topk_weights_desc->dim(1) == topk, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(buf_x_sf_desc->dim(0) == padded && buf_x_sf_desc->dim(1) >= hidden / 128, INFINI_STATUS_BAD_TENSOR_SHAPE);
    *info = Info{tokens, padded, hidden, topk, buf_x_sf_desc->dim(1)};
    return INFINI_STATUS_SUCCESS;
}
} // namespace op::dsv4_sglang_mega_moe_pre_dispatch
#endif
