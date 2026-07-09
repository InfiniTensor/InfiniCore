#ifndef DSV4_MHC_PRE_INFO_H
#define DSV4_MHC_PRE_INFO_H

#include "../../../utils.h"
#include "../../tensor.h"

namespace op::dsv4_mhc_pre {

struct Info {
    infiniDtype_t dtype;
    size_t rows;
    size_t cols;
    float eps;
};

inline size_t leadingRows(infiniopTensorDescriptor_t desc) {
    size_t rows = 1;
    for (size_t i = 0; i + 1 < desc->ndim(); ++i) {
        rows *= desc->dim(i);
    }
    return rows;
}

inline infiniStatus_t createInfo(Info *info,
                                 infiniopTensorDescriptor_t output_desc,
                                 infiniopTensorDescriptor_t input_desc,
                                 infiniopTensorDescriptor_t scale_desc,
                                 infiniopTensorDescriptor_t base_desc,
                                 float eps) {
    CHECK_OR_RETURN(info != nullptr && output_desc != nullptr && input_desc != nullptr && scale_desc != nullptr && base_desc != nullptr, INFINI_STATUS_NULL_POINTER);
    CHECK_DTYPE(input_desc->dtype(), INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32);
    CHECK_OR_RETURN(output_desc->dtype() == input_desc->dtype(), INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(scale_desc->dtype() == INFINI_DTYPE_F32 && base_desc->dtype() == INFINI_DTYPE_F32, INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(input_desc->ndim() >= 2 && output_desc->ndim() == input_desc->ndim(), INFINI_STATUS_BAD_TENSOR_SHAPE);
    for (size_t i = 0; i < input_desc->ndim(); ++i) {
        CHECK_OR_RETURN(output_desc->dim(i) == input_desc->dim(i), INFINI_STATUS_BAD_TENSOR_SHAPE);
    }
    CHECK_OR_RETURN(input_desc->isContiguous() && output_desc->isContiguous() && scale_desc->isContiguous() && base_desc->isContiguous(), INFINI_STATUS_BAD_TENSOR_STRIDES);
    size_t cols = input_desc->dim(input_desc->ndim() - 1);
    CHECK_OR_RETURN(scale_desc->numel() >= 1 && base_desc->numel() == cols, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(eps >= 0.0f, INFINI_STATUS_BAD_PARAM);
    *info = Info{input_desc->dtype(), leadingRows(input_desc), cols, eps};
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::dsv4_mhc_pre

#endif
