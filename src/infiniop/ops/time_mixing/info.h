#ifndef __TIME_MIXING_INFO_H__
#define __TIME_MIXING_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <vector>

namespace op::time_mixing {

class TimeMixingInfo {
    TimeMixingInfo() = default;

public:
    infiniDtype_t dtype;
    int B, T, C, H, N;

    static utils::Result<TimeMixingInfo> create(
        infiniopTensorDescriptor_t y_desc,
        infiniopTensorDescriptor_t r_desc,
        infiniopTensorDescriptor_t w_desc,
        infiniopTensorDescriptor_t k_desc,
        infiniopTensorDescriptor_t v_desc,
        infiniopTensorDescriptor_t a_desc,
        infiniopTensorDescriptor_t b_desc) {
        auto dtype = y_desc->dtype();
        auto shape = y_desc->shape();
        CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32);
        CHECK_OR_RETURN(y_desc->ndim() == 3, INFINI_STATUS_BAD_TENSOR_SHAPE);
        int B = shape[0];
        int T = shape[1];
        int C = shape[2];
        int N = 64;
        int H = C / N;
        CHECK_OR_RETURN(C % N == 0, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_SAME_SHAPE(shape, r_desc->shape(), w_desc->shape(), k_desc->shape(), v_desc->shape(), a_desc->shape(), b_desc->shape());
        return utils::Result<TimeMixingInfo>({dtype, B, T, C, H, N});
    }
};
} // namespace op::time_mixing

#endif // __TIME_MIXING_INFO_H__
