#ifndef DSV4_SGLANG_FUSED_NORM_ROPE_INFO_H
#define DSV4_SGLANG_FUSED_NORM_ROPE_INFO_H
#include "../../../utils.h"
#include "../../tensor.h"
namespace op::dsv4_sglang_fused_norm_rope {
struct Info {
    size_t tokens;
    size_t freqs_rows;
    double eps;
};
inline infiniStatus_t createInfo(Info *info, infiniopTensorDescriptor_t kv_desc, infiniopTensorDescriptor_t weight_desc, infiniopTensorDescriptor_t positions_desc, infiniopTensorDescriptor_t freqs_desc, double eps) {
    CHECK_OR_RETURN(info && kv_desc && weight_desc && positions_desc && freqs_desc, INFINI_STATUS_NULL_POINTER);
    CHECK_OR_RETURN(kv_desc->dtype() == INFINI_DTYPE_BF16 && weight_desc->dtype() == INFINI_DTYPE_BF16 && positions_desc->dtype() == INFINI_DTYPE_I64 && freqs_desc->dtype() == INFINI_DTYPE_F32, INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(kv_desc->ndim() == 2 && weight_desc->ndim() == 1 && positions_desc->ndim() == 1 && freqs_desc->ndim() == 2, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(kv_desc->isContiguous() && weight_desc->isContiguous() && positions_desc->isContiguous() && freqs_desc->isContiguous(), INFINI_STATUS_BAD_TENSOR_STRIDES);
    CHECK_OR_RETURN(kv_desc->dim(1) == 512 && weight_desc->dim(0) == 512 && positions_desc->dim(0) == kv_desc->dim(0) && freqs_desc->dim(1) == 64, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(eps >= 0.0, INFINI_STATUS_BAD_PARAM);
    *info = Info{kv_desc->dim(0), freqs_desc->dim(0), eps};
    return INFINI_STATUS_SUCCESS;
}
} // namespace op::dsv4_sglang_fused_norm_rope
#endif
