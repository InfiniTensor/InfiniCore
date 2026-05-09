#ifndef __BLAS_COPY_INFO_H__
#define __BLAS_COPY_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"

class BlasCopyInfo {
private:
    BlasCopyInfo() = default;

public:
    size_t n;
    ptrdiff_t incx;
    ptrdiff_t incy;
    infiniDtype_t data_type;

    static utils::Result<BlasCopyInfo> createBlasCopyInfo(
        infiniopTensorDescriptor_t x_desc,
        infiniopTensorDescriptor_t y_desc) {

        CHECK_OR_RETURN(x_desc != nullptr, INFINI_STATUS_NULL_POINTER);
        CHECK_OR_RETURN(y_desc != nullptr, INFINI_STATUS_NULL_POINTER);

        auto data_type = x_desc->dtype();

        CHECK_OR_RETURN(y_desc->dtype() == data_type, INFINI_STATUS_BAD_TENSOR_DTYPE);
        CHECK_DTYPE(data_type, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32, INFINI_DTYPE_F64);

        CHECK_OR_RETURN(x_desc->ndim() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(y_desc->ndim() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(x_desc->numel() == y_desc->numel(), INFINI_STATUS_BAD_TENSOR_SHAPE);

        auto n = x_desc->numel();
        auto incx = x_desc->stride(0);
        auto incy = y_desc->stride(0);

        return utils::Result<BlasCopyInfo>(BlasCopyInfo{
            n,
            incx,
            incy,
            data_type});
    }
};

#endif // __BLAS_COPY_INFO_H__
