#ifndef DSV4_SILU_MUL_MASKED_QUANT_INFO_H
#define DSV4_SILU_MUL_MASKED_QUANT_INFO_H

#include "../../../utils.h"
#include "../../tensor.h"

namespace op::dsv4_silu_mul_masked_quant {

struct Info {
    infiniDtype_t dtype;
    size_t rows;
    size_t cols;
    bool has_mask;
};

inline size_t leadingRows(infiniopTensorDescriptor_t desc) {
    size_t rows = 1;
    for (size_t i = 0; i + 1 < desc->ndim(); ++i) {
        rows *= desc->dim(i);
    }
    return rows;
}

inline infiniStatus_t createInfo(Info *info,
                                 infiniopTensorDescriptor_t q_desc,
                                 infiniopTensorDescriptor_t scale_desc,
                                 infiniopTensorDescriptor_t gate_desc,
                                 infiniopTensorDescriptor_t up_desc,
                                 infiniopTensorDescriptor_t mask_desc) {
    CHECK_OR_RETURN(info != nullptr && q_desc != nullptr && scale_desc != nullptr && gate_desc != nullptr && up_desc != nullptr, INFINI_STATUS_NULL_POINTER);
    CHECK_DTYPE(gate_desc->dtype(), INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32);
    CHECK_OR_RETURN(up_desc->dtype() == gate_desc->dtype(), INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(q_desc->dtype() == INFINI_DTYPE_I8 && scale_desc->dtype() == INFINI_DTYPE_F32, INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(gate_desc->ndim() >= 2 && up_desc->ndim() == gate_desc->ndim() && q_desc->ndim() == gate_desc->ndim(), INFINI_STATUS_BAD_TENSOR_SHAPE);
    for (size_t i = 0; i < gate_desc->ndim(); ++i) {
        CHECK_OR_RETURN(up_desc->dim(i) == gate_desc->dim(i) && q_desc->dim(i) == gate_desc->dim(i), INFINI_STATUS_BAD_TENSOR_SHAPE);
    }
    CHECK_OR_RETURN(gate_desc->isContiguous() && up_desc->isContiguous() && q_desc->isContiguous() && scale_desc->isContiguous(), INFINI_STATUS_BAD_TENSOR_STRIDES);
    size_t rows = leadingRows(gate_desc);
    CHECK_OR_RETURN(scale_desc->numel() == rows, INFINI_STATUS_BAD_TENSOR_SHAPE);
    bool has_mask = mask_desc != nullptr;
    if (has_mask) {
        CHECK_OR_RETURN(mask_desc->dtype() == INFINI_DTYPE_I32, INFINI_STATUS_BAD_TENSOR_DTYPE);
        CHECK_OR_RETURN(mask_desc->numel() == rows && mask_desc->isContiguous(), INFINI_STATUS_BAD_TENSOR_SHAPE);
    }
    *info = Info{gate_desc->dtype(), rows, gate_desc->dim(gate_desc->ndim() - 1), has_mask};
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::dsv4_silu_mul_masked_quant

#endif
