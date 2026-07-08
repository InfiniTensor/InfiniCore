#ifndef DSV4_TOPK_TRANSFORM_INFO_H
#define DSV4_TOPK_TRANSFORM_INFO_H
#include "../../../utils.h"
#include "../../tensor.h"
namespace op::dsv4_topk_transform {
struct Info {
    size_t batch;
    size_t index_topk;
    int page_size;
};
inline infiniStatus_t createInfo(Info *info, infiniopTensorDescriptor_t out, infiniopTensorDescriptor_t scores, infiniopTensorDescriptor_t seq_lens, infiniopTensorDescriptor_t page_tables, int page_size) {
    CHECK_OR_RETURN(info && out && scores && seq_lens && page_tables, INFINI_STATUS_NULL_POINTER);
    CHECK_OR_RETURN(out->dtype() == INFINI_DTYPE_I32 && seq_lens->dtype() == INFINI_DTYPE_I32 && page_tables->dtype() == INFINI_DTYPE_I32, INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(scores->dtype() == INFINI_DTYPE_F32, INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(out->ndim() == 2 && scores->ndim() == 2 && seq_lens->ndim() == 1 && page_tables->ndim() >= 2, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(out->dim(0) == scores->dim(0) && out->dim(0) == seq_lens->dim(0) && out->dim(0) == page_tables->dim(0), INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(scores->dim(1) == out->dim(1) * 64, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(out->isContiguous() && scores->isContiguous() && seq_lens->isContiguous() && page_tables->isContiguous(), INFINI_STATUS_BAD_TENSOR_STRIDES);
    *info = Info{out->dim(0), out->dim(1), page_size};
    return INFINI_STATUS_SUCCESS;
}
} // namespace op::dsv4_topk_transform
#endif
