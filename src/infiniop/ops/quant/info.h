#ifndef __QUANT_INFO_H__
#define __QUANT_INFO_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"

namespace op::quant {

class QuantInfo {
private:
    QuantInfo() = default;

public:
    infiniDtype_t dtype, packed_type;
    size_t M, K;

    static utils::Result<QuantInfo> createQuantInfo(
        infiniopTensorDescriptor_t x_packed_desc,
        infiniopTensorDescriptor_t x_scale_desc,
        infiniopTensorDescriptor_t x_zero_desc,
        infiniopTensorDescriptor_t x_desc) {

        CHECK_OR_RETURN(
            x_packed_desc != nullptr && x_scale_desc != nullptr && x_desc != nullptr,
            INFINI_STATUS_NULL_POINTER);

        const infiniDtype_t dtype = x_desc->dtype();
        const infiniDtype_t packed_type = x_packed_desc->dtype();
        CHECK_OR_RETURN(dtype == x_scale_desc->dtype(),
                        INFINI_STATUS_BAD_TENSOR_DTYPE);
        CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32);
        CHECK_DTYPE(packed_type, INFINI_DTYPE_I8);

        CHECK_OR_RETURN(x_desc->ndim() == 2
                            && x_packed_desc->ndim() == 2
                            && x_scale_desc->ndim() == 2,
                        INFINI_STATUS_BAD_TENSOR_SHAPE);

        size_t M = x_desc->dim(0);
        size_t K = x_desc->dim(1);

        CHECK_OR_RETURN(M == x_packed_desc->dim(0)
                            || K == x_packed_desc->dim(1)
                            || M == x_scale_desc->dim(0)
                            || 1 == x_scale_desc->dim(1),
                        INFINI_STATUS_BAD_TENSOR_SHAPE);

        return utils::Result<QuantInfo>(QuantInfo{
            dtype,
            packed_type,
            M,
            K,
        });
    }
};

} // namespace op::quant

#endif //  __QUANT_INFO_H__
