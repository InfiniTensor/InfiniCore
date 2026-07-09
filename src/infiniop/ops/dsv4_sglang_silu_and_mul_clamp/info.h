#ifndef DSV4_SGLANG_SILU_AND_MUL_CLAMP_INFO_H
#define DSV4_SGLANG_SILU_AND_MUL_CLAMP_INFO_H
#include "../../../utils.h"
#include "../../tensor.h"
namespace op::dsv4_sglang_silu_and_mul_clamp {
struct Info {
    size_t tokens;
    size_t hidden;
    double limit;
};
inline infiniStatus_t createInfo(Info *info, infiniopTensorDescriptor_t output_desc, infiniopTensorDescriptor_t input_desc, double limit) {
    CHECK_OR_RETURN(info != nullptr && output_desc != nullptr && input_desc != nullptr, INFINI_STATUS_NULL_POINTER);
    CHECK_OR_RETURN(output_desc->dtype() == INFINI_DTYPE_BF16 && input_desc->dtype() == INFINI_DTYPE_BF16, INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(output_desc->ndim() == 2 && input_desc->ndim() == 2, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(output_desc->isContiguous() && input_desc->isContiguous(), INFINI_STATUS_BAD_TENSOR_STRIDES);
    CHECK_OR_RETURN(output_desc->dim(0) == input_desc->dim(0) && output_desc->dim(1) * 2 == input_desc->dim(1), INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(limit > 0.0, INFINI_STATUS_BAD_PARAM);
    *info = Info{output_desc->dim(0), output_desc->dim(1), limit};
    return INFINI_STATUS_SUCCESS;
}
} // namespace op::dsv4_sglang_silu_and_mul_clamp
#endif
