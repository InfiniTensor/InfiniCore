#ifndef DSV4_SGLANG_FUSED_ROPE_INFO_H
#define DSV4_SGLANG_FUSED_ROPE_INFO_H

#include "../../../utils.h"
#include "../../tensor.h"

namespace op::dsv4_sglang_fused_rope {

struct Info {
    size_t tokens;
    size_t heads;
    bool inverse;
};

inline infiniStatus_t createInfo(Info *info,
                                 infiniopTensorDescriptor_t q_desc,
                                 infiniopTensorDescriptor_t freqs_cis_desc,
                                 infiniopTensorDescriptor_t positions_desc,
                                 bool inverse) {
    CHECK_OR_RETURN(info != nullptr && q_desc != nullptr && freqs_cis_desc != nullptr && positions_desc != nullptr, INFINI_STATUS_NULL_POINTER);
    CHECK_OR_RETURN(q_desc->dtype() == INFINI_DTYPE_BF16 && freqs_cis_desc->dtype() == INFINI_DTYPE_F32 && positions_desc->dtype() == INFINI_DTYPE_I64, INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(q_desc->ndim() == 3 && freqs_cis_desc->ndim() == 2 && positions_desc->ndim() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(q_desc->isContiguous() && freqs_cis_desc->isContiguous() && positions_desc->isContiguous(), INFINI_STATUS_BAD_TENSOR_STRIDES);
    CHECK_OR_RETURN(q_desc->dim(2) == 64, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(freqs_cis_desc->dim(0) >= q_desc->dim(0) && freqs_cis_desc->dim(1) == 64 && positions_desc->dim(0) == q_desc->dim(0), INFINI_STATUS_BAD_TENSOR_SHAPE);

    *info = Info{q_desc->dim(0), q_desc->dim(1), inverse};
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::dsv4_sglang_fused_rope

#endif
