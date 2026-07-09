#ifndef DSV4_SGLANG_TOPK_TRANSFORM_INFO_H
#define DSV4_SGLANG_TOPK_TRANSFORM_INFO_H

#include "../../../utils.h"
#include "../../tensor.h"

namespace op::dsv4_sglang_topk_transform {

struct Info {
    size_t batch;
    size_t scores_width;
    size_t pages;
    size_t topk_width;
    int64_t page_size;
};

inline infiniStatus_t createInfo(Info *info,
                                 infiniopTensorDescriptor_t scores_desc,
                                 infiniopTensorDescriptor_t seq_lens_desc,
                                 infiniopTensorDescriptor_t page_table_desc,
                                 infiniopTensorDescriptor_t page_indices_desc,
                                 infiniopTensorDescriptor_t raw_indices_desc,
                                 int64_t page_size) {
    CHECK_OR_RETURN(info != nullptr && scores_desc != nullptr && seq_lens_desc != nullptr && page_table_desc != nullptr && page_indices_desc != nullptr && raw_indices_desc != nullptr, INFINI_STATUS_NULL_POINTER);
    CHECK_OR_RETURN(scores_desc->dtype() == INFINI_DTYPE_F32 && seq_lens_desc->dtype() == INFINI_DTYPE_I32 && page_table_desc->dtype() == INFINI_DTYPE_I32 && page_indices_desc->dtype() == INFINI_DTYPE_I32 && raw_indices_desc->dtype() == INFINI_DTYPE_I32, INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(scores_desc->ndim() == 2 && page_table_desc->ndim() == 2 && page_indices_desc->ndim() == 2 && raw_indices_desc->ndim() == 2 && seq_lens_desc->ndim() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(scores_desc->isContiguous() && seq_lens_desc->isContiguous() && page_table_desc->isContiguous() && page_indices_desc->isContiguous() && raw_indices_desc->isContiguous(), INFINI_STATUS_BAD_TENSOR_STRIDES);
    CHECK_OR_RETURN(page_size > 0, INFINI_STATUS_BAD_PARAM);

    size_t batch = scores_desc->dim(0);
    size_t width = page_indices_desc->dim(1);
    CHECK_OR_RETURN(batch > 0 && (width == 512 || width == 1024), INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(seq_lens_desc->dim(0) == batch && page_table_desc->dim(0) == batch && page_indices_desc->dim(0) == batch && raw_indices_desc->dim(0) == batch && raw_indices_desc->dim(1) == width, INFINI_STATUS_BAD_TENSOR_SHAPE);

    *info = Info{batch, scores_desc->dim(1), page_table_desc->dim(1), width, page_size};
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::dsv4_sglang_topk_transform

#endif
