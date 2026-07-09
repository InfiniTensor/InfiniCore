#ifndef DSV4_SGLANG_TOPK_V2_INFO_H
#define DSV4_SGLANG_TOPK_V2_INFO_H
#include "../../../utils.h"
#include "../../tensor.h"
namespace op::dsv4_sglang_topk_v2 {
struct Info {
    size_t batch;
    size_t pages;
    size_t topk_width;
    size_t workspace_width;
    size_t metadata_rows;
    int64_t page_size;
};
inline infiniStatus_t createInfo(Info *info, infiniopTensorDescriptor_t scores_desc, infiniopTensorDescriptor_t seq_lens_desc, infiniopTensorDescriptor_t page_table_desc, infiniopTensorDescriptor_t page_indices_desc, infiniopTensorDescriptor_t transform_workspace_desc, infiniopTensorDescriptor_t metadata_desc, int64_t page_size) {
    CHECK_OR_RETURN(info && scores_desc && seq_lens_desc && page_table_desc && page_indices_desc && transform_workspace_desc && metadata_desc, INFINI_STATUS_NULL_POINTER);
    CHECK_OR_RETURN(scores_desc->dtype() == INFINI_DTYPE_F32 && seq_lens_desc->dtype() == INFINI_DTYPE_I32 && page_table_desc->dtype() == INFINI_DTYPE_I32 && page_indices_desc->dtype() == INFINI_DTYPE_I32 && transform_workspace_desc->dtype() == INFINI_DTYPE_I32 && metadata_desc->dtype() == INFINI_DTYPE_I32, INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(scores_desc->ndim() == 2 && seq_lens_desc->ndim() == 1 && page_table_desc->ndim() == 2 && page_indices_desc->ndim() == 2 && transform_workspace_desc->ndim() == 2 && metadata_desc->ndim() == 2, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(scores_desc->isContiguous() && seq_lens_desc->isContiguous() && page_table_desc->isContiguous() && page_indices_desc->isContiguous() && transform_workspace_desc->isContiguous() && metadata_desc->isContiguous(), INFINI_STATUS_BAD_TENSOR_STRIDES);
    size_t batch = scores_desc->dim(0);
    size_t width = page_indices_desc->dim(1);
    CHECK_OR_RETURN(batch > 0 && scores_desc->dim(1) == 512 && width == 512, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(seq_lens_desc->dim(0) == batch && page_table_desc->dim(0) == batch && page_indices_desc->dim(0) == batch && transform_workspace_desc->dim(0) == batch, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(transform_workspace_desc->dim(1) >= 2050 && metadata_desc->dim(0) == batch + 1 && metadata_desc->dim(1) == 4, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(page_size > 0, INFINI_STATUS_BAD_PARAM);
    *info = Info{batch, page_table_desc->dim(1), width, transform_workspace_desc->dim(1), metadata_desc->dim(0), page_size};
    return INFINI_STATUS_SUCCESS;
}
} // namespace op::dsv4_sglang_topk_v2
#endif
