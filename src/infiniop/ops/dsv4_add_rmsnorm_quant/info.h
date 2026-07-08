#ifndef DSV4_ADD_RMSNORM_QUANT_INFO_H
#define DSV4_ADD_RMSNORM_QUANT_INFO_H
#include "../../../utils.h"
#include "../../tensor.h"
namespace op::dsv4_add_rmsnorm_quant {
struct Info {
    size_t rows, cols;
    infiniDtype_t dtype;
};
inline infiniStatus_t createInfo(Info *info, infiniopTensorDescriptor_t res, infiniopTensorDescriptor_t q, infiniopTensorDescriptor_t scale, infiniopTensorDescriptor_t x, infiniopTensorDescriptor_t weight) {
    CHECK_OR_RETURN(info && res && q && scale && x && weight, INFINI_STATUS_NULL_POINTER);
    CHECK_OR_RETURN(q->dtype() == INFINI_DTYPE_I8 && scale->dtype() == INFINI_DTYPE_F32, INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(res->dtype() == x->dtype() && res->dtype() == weight->dtype() && (res->dtype() == INFINI_DTYPE_BF16 || res->dtype() == INFINI_DTYPE_F16), INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(res->ndim() == x->ndim() && q->ndim() == res->ndim() && res->ndim() >= 2 && weight->ndim() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(res->shape() == x->shape() && q->shape() == res->shape(), INFINI_STATUS_BAD_TENSOR_SHAPE);
    size_t rows = res->numel() / res->dim(res->ndim() - 1);
    size_t cols = res->dim(res->ndim() - 1);
    CHECK_OR_RETURN(weight->dim(0) == cols && scale->numel() == rows, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(res->isContiguous() && q->isContiguous() && scale->isContiguous() && x->isContiguous() && weight->isContiguous(), INFINI_STATUS_BAD_TENSOR_STRIDES);
    *info = Info{rows, cols, res->dtype()};
    return INFINI_STATUS_SUCCESS;
}
} // namespace op::dsv4_add_rmsnorm_quant
#endif
