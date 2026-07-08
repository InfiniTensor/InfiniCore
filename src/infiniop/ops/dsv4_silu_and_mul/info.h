#ifndef DSV4_SILU_AND_MUL_INFO_H
#define DSV4_SILU_AND_MUL_INFO_H
#include "../../../utils.h"
#include "../../tensor.h"
namespace op::dsv4_silu_and_mul {
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
inline infiniStatus_t createInfo(Info *info, infiniopTensorDescriptor_t y, infiniopTensorDescriptor_t gate, infiniopTensorDescriptor_t up) {
    CHECK_OR_RETURN(info && y && gate && up, INFINI_STATUS_NULL_POINTER);
    CHECK_DTYPE(gate->dtype(), INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32);
    CHECK_OR_RETURN(y->dtype() == gate->dtype() && up->dtype() == gate->dtype(), INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(gate->ndim() >= 1 && y->ndim() == gate->ndim() && up->ndim() == gate->ndim(), INFINI_STATUS_BAD_TENSOR_SHAPE);
    for (size_t i = 0; i < gate->ndim(); ++i) {
        CHECK_OR_RETURN(y->dim(i) == gate->dim(i) && up->dim(i) == gate->dim(i), INFINI_STATUS_BAD_TENSOR_SHAPE);
    }
    CHECK_OR_RETURN(y->isContiguous() && gate->isContiguous() && up->isContiguous(), INFINI_STATUS_BAD_TENSOR_STRIDES);
    *info = Info{gate->dtype(), leadingRows(gate), gate->dim(gate->ndim() - 1)};
    return INFINI_STATUS_SUCCESS;
}
} // namespace op::dsv4_silu_and_mul
#endif
