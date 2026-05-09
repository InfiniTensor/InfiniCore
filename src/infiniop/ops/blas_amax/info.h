#ifndef __BLAS_AMAX_INFO_H__
#define __BLAS_AMAX_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"

class BlasAmaxInfo {
private:
    BlasAmaxInfo() = default;

public:
    size_t n;
    ptrdiff_t incx;
    infiniDtype_t data_type;

    static utils::Result<BlasAmaxInfo> createBlasAmaxInfo(
        infiniopTensorDescriptor_t x_desc,
        infiniopTensorDescriptor_t result_desc) {

        CHECK_OR_RETURN(x_desc != nullptr, INFINI_STATUS_NULL_POINTER);
        CHECK_OR_RETURN(result_desc != nullptr, INFINI_STATUS_NULL_POINTER);

        auto data_type = x_desc->dtype();
        auto itype = result_desc->dtype();
        CHECK_DTYPE(data_type, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32, INFINI_DTYPE_F64);
        CHECK_DTYPE(itype, INFINI_DTYPE_I32);

        CHECK_OR_RETURN(x_desc->ndim() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(result_desc->numel() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);

        auto n = x_desc->numel();
        auto incx = x_desc->stride(0);

        return utils::Result<BlasAmaxInfo>(BlasAmaxInfo{
            n,
            incx,
            data_type});
    }
};

#endif // __BLAS_AMAX_INFO_H__
