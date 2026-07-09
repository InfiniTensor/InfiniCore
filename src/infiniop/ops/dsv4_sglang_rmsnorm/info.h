#ifndef DSV4_SGLANG_RMSNORM_INFO_H
#define DSV4_SGLANG_RMSNORM_INFO_H

#include "../../../utils.h"
#include "../../tensor.h"

namespace op::dsv4_sglang_rmsnorm {

struct Info {
    size_t batch;
    size_t tokens;
    double eps;
};

inline infiniStatus_t createInfo(Info *info,
                                 infiniopTensorDescriptor_t output_desc,
                                 infiniopTensorDescriptor_t input_desc,
                                 double eps) {
    CHECK_OR_RETURN(info != nullptr && output_desc != nullptr && input_desc != nullptr, INFINI_STATUS_NULL_POINTER);
    CHECK_OR_RETURN(output_desc->dtype() == INFINI_DTYPE_BF16 && input_desc->dtype() == INFINI_DTYPE_BF16, INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(output_desc->ndim() == 3 && input_desc->ndim() == 3, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(output_desc->isContiguous() && input_desc->isContiguous(), INFINI_STATUS_BAD_TENSOR_STRIDES);
    CHECK_OR_RETURN(input_desc->dim(0) == output_desc->dim(0) && input_desc->dim(1) == output_desc->dim(1) && input_desc->dim(2) == output_desc->dim(2), INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(input_desc->dim(2) == 512, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(eps >= 0.0, INFINI_STATUS_BAD_PARAM);

    *info = Info{input_desc->dim(0), input_desc->dim(1), eps};
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::dsv4_sglang_rmsnorm

#endif
