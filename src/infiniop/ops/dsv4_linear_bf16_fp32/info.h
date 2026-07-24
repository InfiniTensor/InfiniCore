#ifndef DSV4_LINEAR_BF16_FP32_INFO_H
#define DSV4_LINEAR_BF16_FP32_INFO_H

#include "../../../utils.h"
#include "../../tensor.h"

namespace op::dsv4_linear_bf16_fp32 {

struct Info {
    size_t m;
    size_t n;
    size_t k;
    infiniDtype_t x_dtype;
};

inline infiniStatus_t createInfo(
    Info *info,
    infiniopTensorDescriptor_t y,
    infiniopTensorDescriptor_t x,
    infiniopTensorDescriptor_t w) {
    CHECK_OR_RETURN(info && y && x && w, INFINI_STATUS_NULL_POINTER);
    CHECK_OR_RETURN(y->dtype() == INFINI_DTYPE_F32, INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(w->dtype() == INFINI_DTYPE_F32, INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(x->dtype() == INFINI_DTYPE_BF16 || x->dtype() == INFINI_DTYPE_F16, INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(y->ndim() == 2 && x->ndim() == 2 && w->ndim() == 2, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(y->dim(0) == x->dim(0), INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(y->dim(1) == w->dim(0), INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(x->dim(1) == w->dim(1), INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(y->isContiguous() && x->isContiguous() && w->isContiguous(), INFINI_STATUS_BAD_TENSOR_STRIDES);
    *info = Info{y->dim(0), y->dim(1), x->dim(1), x->dtype()};
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::dsv4_linear_bf16_fp32

#endif
