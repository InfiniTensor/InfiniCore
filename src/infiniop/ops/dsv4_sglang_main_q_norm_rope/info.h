#ifndef DSV4_SGLANG_MAIN_Q_NORM_ROPE_INFO_H
#define DSV4_SGLANG_MAIN_Q_NORM_ROPE_INFO_H
#include "../../../utils.h"
#include "../../tensor.h"
namespace op::dsv4_sglang_main_q_norm_rope {
struct Info {
    size_t batch;
    size_t heads;
    size_t freqs_rows;
    double eps;
};
inline infiniStatus_t createInfo(Info *info, infiniopTensorDescriptor_t output_desc, infiniopTensorDescriptor_t input_desc, infiniopTensorDescriptor_t freqs_desc, infiniopTensorDescriptor_t positions_desc, double eps) {
    CHECK_OR_RETURN(info && output_desc && input_desc && freqs_desc && positions_desc, INFINI_STATUS_NULL_POINTER);
    CHECK_OR_RETURN(output_desc->dtype() == INFINI_DTYPE_BF16 && input_desc->dtype() == INFINI_DTYPE_BF16 && freqs_desc->dtype() == INFINI_DTYPE_F32 && positions_desc->dtype() == INFINI_DTYPE_I32, INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(output_desc->ndim() == 3 && input_desc->ndim() == 3 && freqs_desc->ndim() == 2 && positions_desc->ndim() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(output_desc->isContiguous() && input_desc->isContiguous() && freqs_desc->isContiguous() && positions_desc->isContiguous(), INFINI_STATUS_BAD_TENSOR_STRIDES);
    CHECK_OR_RETURN(output_desc->dim(0) == input_desc->dim(0) && output_desc->dim(1) == input_desc->dim(1) && output_desc->dim(2) == input_desc->dim(2), INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(input_desc->dim(2) == 512 && positions_desc->dim(0) == input_desc->dim(0) && freqs_desc->dim(1) == 64, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(eps >= 0.0, INFINI_STATUS_BAD_PARAM);
    *info = Info{input_desc->dim(0), input_desc->dim(1), freqs_desc->dim(0), eps};
    return INFINI_STATUS_SUCCESS;
}
} // namespace op::dsv4_sglang_main_q_norm_rope
#endif
