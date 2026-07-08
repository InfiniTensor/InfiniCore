#ifndef DSV4_ACT_QUANT_FP8_INFO_H
#define DSV4_ACT_QUANT_FP8_INFO_H
#include "../../../utils.h"
#include "../../tensor.h"
namespace op::dsv4_act_quant_fp8 {
struct Info {
    size_t rows, cols;
    infiniDtype_t dtype;
    float fp8_max;
};
inline infiniStatus_t createInfo(Info *info, infiniopTensorDescriptor_t xq, infiniopTensorDescriptor_t scale, infiniopTensorDescriptor_t x, float fp8_max) {
    CHECK_OR_RETURN(info && xq && scale && x, INFINI_STATUS_NULL_POINTER);
    CHECK_OR_RETURN(xq->dtype() == INFINI_DTYPE_F8 && scale->dtype() == INFINI_DTYPE_F32, INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(x->dtype() == INFINI_DTYPE_BF16 || x->dtype() == INFINI_DTYPE_F16, INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(xq->ndim() == x->ndim() && x->ndim() >= 2 && xq->shape() == x->shape(), INFINI_STATUS_BAD_TENSOR_SHAPE);
    size_t rows = x->numel() / x->dim(x->ndim() - 1);
    size_t cols = x->dim(x->ndim() - 1);
    CHECK_OR_RETURN(scale->numel() == rows, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(xq->isContiguous() && scale->isContiguous() && x->isContiguous(), INFINI_STATUS_BAD_TENSOR_STRIDES);
    *info = Info{rows, cols, x->dtype(), fp8_max};
    return INFINI_STATUS_SUCCESS;
}
} // namespace op::dsv4_act_quant_fp8
#endif
