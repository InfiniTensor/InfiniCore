#ifndef DSV4_FUSED_ROPE_INFO_H
#define DSV4_FUSED_ROPE_INFO_H
#include "../../../utils.h"
#include "../../tensor.h"
namespace op::dsv4_fused_rope {
struct Info {
    infiniDtype_t dtype;
    size_t seq_len;
    size_t q_heads;
    size_t k_heads;
    size_t rope_dim;
    bool has_k;
};
inline infiniStatus_t createInfo(Info *info, infiniopTensorDescriptor_t q, infiniopTensorDescriptor_t k, infiniopTensorDescriptor_t fr, infiniopTensorDescriptor_t fi, int has_k) {
    CHECK_OR_RETURN(info && q && fr && fi, INFINI_STATUS_NULL_POINTER);
    CHECK_DTYPE(q->dtype(), INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32);
    CHECK_OR_RETURN(fr->dtype() == INFINI_DTYPE_F32 && fi->dtype() == INFINI_DTYPE_F32, INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(q->ndim() == 3 && fr->ndim() == 2 && fi->ndim() == 2, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(q->dim(2) % 2 == 0, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(fr->dim(0) == q->dim(0) && fi->dim(0) == q->dim(0) && fr->dim(1) == q->dim(2) / 2 && fi->dim(1) == q->dim(2) / 2, INFINI_STATUS_BAD_TENSOR_SHAPE);
    size_t k_heads = 0;
    if (has_k) {
        CHECK_OR_RETURN(k != nullptr, INFINI_STATUS_NULL_POINTER);
        CHECK_OR_RETURN(k->dtype() == q->dtype() && k->ndim() == 3 && k->dim(0) == q->dim(0) && k->dim(2) == q->dim(2), INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(k->isContiguous(), INFINI_STATUS_BAD_TENSOR_STRIDES);
        k_heads = k->dim(1);
    }
    CHECK_OR_RETURN(q->isContiguous() && fr->isContiguous() && fi->isContiguous(), INFINI_STATUS_BAD_TENSOR_STRIDES);
    *info = Info{q->dtype(), q->dim(0), q->dim(1), k_heads, q->dim(2), has_k != 0};
    return INFINI_STATUS_SUCCESS;
}
} // namespace op::dsv4_fused_rope
#endif
