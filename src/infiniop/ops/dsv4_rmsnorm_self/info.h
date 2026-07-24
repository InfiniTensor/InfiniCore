#ifndef DSV4_RMSNORM_SELF_INFO_H
#define DSV4_RMSNORM_SELF_INFO_H
#include "../../../utils.h"
#include "../../tensor.h"
namespace op::dsv4_rmsnorm_self {
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
inline infiniStatus_t createInfo(Info *info, infiniopTensorDescriptor_t y_desc, infiniopTensorDescriptor_t x_desc) {
    CHECK_OR_RETURN(info && y_desc && x_desc, INFINI_STATUS_NULL_POINTER);
    CHECK_DTYPE(x_desc->dtype(), INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32);
    CHECK_OR_RETURN(y_desc->dtype() == x_desc->dtype(), INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(x_desc->ndim() >= 2 && y_desc->ndim() == x_desc->ndim(), INFINI_STATUS_BAD_TENSOR_SHAPE);
    for (size_t i = 0; i < x_desc->ndim(); ++i) {
        CHECK_OR_RETURN(y_desc->dim(i) == x_desc->dim(i), INFINI_STATUS_BAD_TENSOR_SHAPE);
    }
    CHECK_OR_RETURN(x_desc->isContiguous() && y_desc->isContiguous(), INFINI_STATUS_BAD_TENSOR_STRIDES);
    *info = Info{x_desc->dtype(), leadingRows(x_desc), x_desc->dim(x_desc->ndim() - 1)};
    return INFINI_STATUS_SUCCESS;
}
} // namespace op::dsv4_rmsnorm_self
#endif
