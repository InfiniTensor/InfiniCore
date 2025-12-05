#ifndef __LINEAR_INFO_H__
#define __LINEAR_INFO_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"

namespace op::linear {

class LinearInfo {
private:
    LinearInfo() = default;

public:
    infiniDtype_t dtype, packed_type;
    size_t M, K, N;
    float alpha, beta;

    static utils::Result<LinearInfo> createLinearInfo(
        infiniopTensorDescriptor_t d_desc,
        infiniopTensorDescriptor_t c_desc,
        infiniopTensorDescriptor_t bias_desc,
        infiniopTensorDescriptor_t x_desc,
        infiniopTensorDescriptor_t x_scale_desc,
        infiniopTensorDescriptor_t x_zero_desc,
        infiniopTensorDescriptor_t weights_desc,
        infiniopTensorDescriptor_t weights_scale_desc,
        infiniopTensorDescriptor_t weights_zero_desc,
        float alpha,
        float beta) {

        CHECK_OR_RETURN(
            d_desc != nullptr && c_desc != nullptr && bias_desc != nullptr && x_desc != nullptr && x_scale_desc != nullptr && weights_desc != nullptr && weights_scale_desc != nullptr,
            INFINI_STATUS_NULL_POINTER);

        const infiniDtype_t dtype = d_desc->dtype();
        const infiniDtype_t packed_type = x_desc->dtype();
        CHECK_OR_RETURN(dtype == c_desc->dtype() && dtype == bias_desc->dtype() && dtype == x_scale_desc->dtype() && dtype == weights_scale_desc->dtype(),
                        INFINI_STATUS_BAD_TENSOR_DTYPE);
        CHECK_OR_RETURN(packed_type == weights_desc->dtype(),
                        INFINI_STATUS_BAD_TENSOR_DTYPE);
        CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32);
        CHECK_DTYPE(packed_type, INFINI_DTYPE_I8);
        CHECK_OR_RETURN(bias_desc->ndim() == 1,
                        INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(d_desc->ndim() == 2
                            && c_desc->ndim() == 2
                            && x_desc->ndim() == 2
                            && x_scale_desc->ndim() == 2
                            && weights_desc->ndim() == 2
                            && weights_scale_desc->ndim() == 2,
                        INFINI_STATUS_BAD_TENSOR_SHAPE);

        size_t M = d_desc->dim(0);
        size_t N = d_desc->dim(1);
        size_t K = x_desc->dim(1);
        CHECK_OR_RETURN(N == bias_desc->dim(0),
                        INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(M == x_desc->dim(0)
                            || M == x_scale_desc->dim(0)
                            || 1 == x_scale_desc->dim(1)
                            || 1 == weights_scale_desc->dim(0)
                            || N == weights_scale_desc->dim(1),
                        INFINI_STATUS_BAD_TENSOR_SHAPE);

        return utils::Result<LinearInfo>(LinearInfo{
            dtype,
            packed_type,
            M,
            K,
            N,
            alpha,
            beta});
    }
};

} // namespace op::linear

#endif //  __LINEAR_INFO_H__
