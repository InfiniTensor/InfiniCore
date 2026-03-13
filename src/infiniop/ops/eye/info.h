#ifndef __EYE_INFO_H__
#define __EYE_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"

namespace op::eye {

struct EyeInfo {
    infiniDtype_t dtype;
    size_t n; // the number of rows
    size_t m; // optional, the number of columns with default being n

    static utils::Result<EyeInfo> create(infiniopTensorDescriptor_t y_desc) {
        if (!y_desc) {
            return INFINI_STATUS_BAD_PARAM;
        }

        auto dtype = y_desc->dtype();
        if (dtype != INFINI_DTYPE_F16 && dtype != INFINI_DTYPE_F32 && dtype != INFINI_DTYPE_F64 && dtype != INFINI_DTYPE_BF16 && dtype != INFINI_DTYPE_I32 && dtype != INFINI_DTYPE_I64) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        if (y_desc->ndim() != 2) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        auto y_shape = y_desc->shape();
        size_t n = y_shape[0];
        size_t m = y_shape[1];

        return utils::Result<EyeInfo>(EyeInfo{dtype, n, m});
    }
};

} // namespace op::eye

#endif
