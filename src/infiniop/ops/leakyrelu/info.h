#ifndef __LEAKYRELU_INFO_H__
#define __LEAKYRELU_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <vector>

namespace op::leakyrelu {

class LeakyReLUInfo {
    LeakyReLUInfo() = default;

public:
    infiniDtype_t dt_in;
    std::vector<size_t> shape;
    std::vector<ptrdiff_t> in_stride;
    std::vector<ptrdiff_t> out_stride;
    size_t n;
    float negative_slope;

    static utils::Result<LeakyReLUInfo> create(
        infiniopTensorDescriptor_t out_desc,
        infiniopTensorDescriptor_t in_desc,
        float negative_slope) {

        auto dt_raw = in_desc->dtype();
        infiniDtype_t dt_in = dt_raw;

        CHECK_DTYPE(dt_in, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32, INFINI_DTYPE_F64);

        CHECK_OR_RETURN(out_desc->ndim() == in_desc->ndim(), INFINI_STATUS_BAD_TENSOR_SHAPE);
        for (size_t i = 0; i < out_desc->ndim(); ++i) {
            CHECK_OR_RETURN(out_desc->dim(i) == in_desc->dim(i), INFINI_STATUS_BAD_TENSOR_SHAPE);
        }

        size_t n = 1;
        for (size_t i = 0; i < in_desc->ndim(); ++i) {
            n *= static_cast<size_t>(in_desc->dim(i));
        }

        return utils::Result<LeakyReLUInfo>(LeakyReLUInfo{
            dt_in,
            out_desc->shape(),
            in_desc->strides(),
            out_desc->strides(),
            n,
            negative_slope});
    }
};

} // namespace op::leakyrelu

#endif // __LEAKYRELU_INFO_H__
