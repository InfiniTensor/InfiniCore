#ifndef __CAST_INFO_H__
#define __CAST_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <vector>

namespace op::cast {

class CastInfo {
    CastInfo() = default;

public:
    infiniDtype_t dt_in;
    infiniDtype_t dt_out;
    std::vector<size_t> shape;
    std::vector<ptrdiff_t> in_stride;
    std::vector<ptrdiff_t> out_stride;
    size_t n;

    static utils::Result<CastInfo> create(
        infiniopTensorDescriptor_t out_desc,
        infiniopTensorDescriptor_t in_desc) {

        auto dt_out = out_desc->dtype();
        auto dt_in = in_desc->dtype();

        CHECK_DTYPE(dt_in,
                    INFINI_DTYPE_I32, INFINI_DTYPE_I64,
                    INFINI_DTYPE_U32, INFINI_DTYPE_U64,
                    INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64);
        CHECK_DTYPE(dt_out,
                    INFINI_DTYPE_I32, INFINI_DTYPE_I64,
                    INFINI_DTYPE_U32, INFINI_DTYPE_U64,
                    INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64);

        CHECK_OR_RETURN(out_desc->ndim() == in_desc->ndim(), INFINI_STATUS_BAD_TENSOR_SHAPE);
        for (size_t i = 0; i < out_desc->ndim(); ++i) {
            CHECK_OR_RETURN(out_desc->dim(i) == in_desc->dim(i), INFINI_STATUS_BAD_TENSOR_SHAPE);
        }

        size_t n = 1;
        for (size_t i = 0; i < in_desc->ndim(); ++i) {
            n *= static_cast<size_t>(in_desc->dim(i));
        }

        return utils::Result<CastInfo>(CastInfo{
            dt_in,
            dt_out,
            out_desc->shape(),
            in_desc->strides(),
            out_desc->strides(),
            n,
        });
    }
};

} // namespace op::cast

#endif // __CAST_INFO_H__
