#ifndef DSV4_DEEPGEMM_TF32_HC_PERNORM_GEMM_INFO_H
#define DSV4_DEEPGEMM_TF32_HC_PERNORM_GEMM_INFO_H

#include "../../../utils.h"
#include "../../tensor.h"

namespace op::dsv4_deepgemm_tf32_hc_pernorm_gemm {

struct Info {
    size_t m;
    size_t n;
    size_t k;
    size_t num_splits;
    int d_ndim;
    int sqr_sum_ndim;
};

inline infiniStatus_t createInfo(Info *info,
                                 infiniopTensorDescriptor_t a_desc,
                                 infiniopTensorDescriptor_t b_desc,
                                 infiniopTensorDescriptor_t d_desc,
                                 infiniopTensorDescriptor_t sqr_sum_desc,
                                 int64_t num_splits) {
    CHECK_OR_RETURN(info != nullptr && a_desc != nullptr && b_desc != nullptr && d_desc != nullptr && sqr_sum_desc != nullptr, INFINI_STATUS_NULL_POINTER);
    CHECK_OR_RETURN(a_desc->dtype() == INFINI_DTYPE_BF16 && b_desc->dtype() == INFINI_DTYPE_F32 && d_desc->dtype() == INFINI_DTYPE_F32 && sqr_sum_desc->dtype() == INFINI_DTYPE_F32,
                    INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(a_desc->ndim() == 2 && b_desc->ndim() == 2, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(d_desc->ndim() == 2 || d_desc->ndim() == 3, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(sqr_sum_desc->ndim() == 1 || sqr_sum_desc->ndim() == 2, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(a_desc->isContiguous() && b_desc->isContiguous() && d_desc->isContiguous() && sqr_sum_desc->isContiguous(), INFINI_STATUS_BAD_TENSOR_STRIDES);
    CHECK_OR_RETURN(num_splits > 0, INFINI_STATUS_BAD_PARAM);

    const auto m = a_desc->dim(0);
    const auto k = a_desc->dim(1);
    const auto n = b_desc->dim(0);
    CHECK_OR_RETURN(m > 0 && n > 0 && k > 0, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(b_desc->dim(1) == k, INFINI_STATUS_BAD_TENSOR_SHAPE);

    if (d_desc->ndim() == 2) {
        CHECK_OR_RETURN(num_splits == 1 && d_desc->dim(0) == m && d_desc->dim(1) == n, INFINI_STATUS_BAD_TENSOR_SHAPE);
    } else {
        CHECK_OR_RETURN(d_desc->dim(0) == static_cast<size_t>(num_splits) && d_desc->dim(1) == m && d_desc->dim(2) == n, INFINI_STATUS_BAD_TENSOR_SHAPE);
    }
    if (sqr_sum_desc->ndim() == 1) {
        CHECK_OR_RETURN(num_splits == 1 && sqr_sum_desc->dim(0) == m, INFINI_STATUS_BAD_TENSOR_SHAPE);
    } else {
        CHECK_OR_RETURN(sqr_sum_desc->dim(0) == static_cast<size_t>(num_splits) && sqr_sum_desc->dim(1) == m, INFINI_STATUS_BAD_TENSOR_SHAPE);
    }

    *info = Info{m, n, k, static_cast<size_t>(num_splits), static_cast<int>(d_desc->ndim()), static_cast<int>(sqr_sum_desc->ndim())};
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::dsv4_deepgemm_tf32_hc_pernorm_gemm

#endif
