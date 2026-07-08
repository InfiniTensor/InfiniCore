#ifndef DSV4_PER_TOKEN_QUANT_INT8_INFO_H
#define DSV4_PER_TOKEN_QUANT_INT8_INFO_H

#include "../../../utils.h"
#include "../../tensor.h"

namespace op::dsv4_per_token_quant_int8 {

struct Info {
    infiniDtype_t dtype;
    size_t rows;
    size_t cols;
};

inline size_t leadingRows(infiniopTensorDescriptor_t desc) {
    size_t rows = 1;
    for (size_t i = 0; i + 1 < desc->ndim(); ++i) {
        rows *= desc->dim(i);
    }
    return rows;
}

inline infiniStatus_t createInfo(Info *info, infiniopTensorDescriptor_t q_desc, infiniopTensorDescriptor_t scale_desc, infiniopTensorDescriptor_t x_desc) {
    CHECK_OR_RETURN(info != nullptr && q_desc != nullptr && scale_desc != nullptr && x_desc != nullptr, INFINI_STATUS_NULL_POINTER);
    CHECK_DTYPE(x_desc->dtype(), INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32);
    CHECK_OR_RETURN(q_desc->dtype() == INFINI_DTYPE_I8 && scale_desc->dtype() == INFINI_DTYPE_F32, INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(x_desc->ndim() >= 2 && q_desc->ndim() == x_desc->ndim(), INFINI_STATUS_BAD_TENSOR_SHAPE);
    for (size_t i = 0; i < x_desc->ndim(); ++i) {
        CHECK_OR_RETURN(q_desc->dim(i) == x_desc->dim(i), INFINI_STATUS_BAD_TENSOR_SHAPE);
    }
    CHECK_OR_RETURN(x_desc->isContiguous() && q_desc->isContiguous() && scale_desc->isContiguous(), INFINI_STATUS_BAD_TENSOR_STRIDES);
    size_t rows = leadingRows(x_desc);
    CHECK_OR_RETURN(scale_desc->numel() == rows, INFINI_STATUS_BAD_TENSOR_SHAPE);
    *info = Info{x_desc->dtype(), rows, x_desc->dim(x_desc->ndim() - 1)};
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::dsv4_per_token_quant_int8

#endif
