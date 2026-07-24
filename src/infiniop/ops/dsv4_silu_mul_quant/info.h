#ifndef DSV4_SILU_MUL_QUANT_INFO_H
#define DSV4_SILU_MUL_QUANT_INFO_H
#include "../../../utils.h"
#include "../../tensor.h"
namespace op::dsv4_silu_mul_quant {
struct Info {
    size_t rows, cols;
    infiniDtype_t dtype;
};
inline infiniStatus_t createInfo(Info *info, infiniopTensorDescriptor_t q, infiniopTensorDescriptor_t scale, infiniopTensorDescriptor_t gate, infiniopTensorDescriptor_t up) {
    CHECK_OR_RETURN(info && q && scale && gate && up, INFINI_STATUS_NULL_POINTER);
    CHECK_OR_RETURN(q->dtype() == INFINI_DTYPE_I8 && scale->dtype() == INFINI_DTYPE_F32, INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(gate->dtype() == up->dtype() && (gate->dtype() == INFINI_DTYPE_BF16 || gate->dtype() == INFINI_DTYPE_F16), INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(q->ndim() == gate->ndim() && gate->ndim() == up->ndim() && gate->ndim() >= 2, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(q->shape() == gate->shape() && up->shape() == gate->shape(), INFINI_STATUS_BAD_TENSOR_SHAPE);
    size_t rows = gate->numel() / gate->dim(gate->ndim() - 1);
    size_t cols = gate->dim(gate->ndim() - 1);
    CHECK_OR_RETURN(scale->numel() == rows, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(q->isContiguous() && scale->isContiguous() && gate->isContiguous() && up->isContiguous(), INFINI_STATUS_BAD_TENSOR_STRIDES);
    *info = Info{rows, cols, gate->dtype()};
    return INFINI_STATUS_SUCCESS;
}
} // namespace op::dsv4_silu_mul_quant
#endif
