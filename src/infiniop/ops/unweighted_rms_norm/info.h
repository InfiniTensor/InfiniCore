#ifndef __UNWEIGHTED_RMS_NORM_INFO_H__
#define __UNWEIGHTED_RMS_NORM_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <numeric>
#include <vector>

namespace op::unweighted_rms_norm {

class UnweightedRMSNormInfo {
    UnweightedRMSNormInfo() = default;

    static bool is_contiguous(infiniopTensorDescriptor_t desc) {
        ptrdiff_t expected = 1;
        for (ptrdiff_t i = static_cast<ptrdiff_t>(desc->ndim()) - 1; i >= 0; --i) {
            if (desc->stride(static_cast<size_t>(i)) != expected) {
                return false;
            }
            expected *= static_cast<ptrdiff_t>(desc->dim(static_cast<size_t>(i)));
        }
        return true;
    }

public:
    infiniDtype_t atype;
    float epsilon;
    std::vector<size_t> shape;
    size_t outer_size;
    size_t last_dim;

    size_t ndim() const { return shape.size(); }
    size_t dim() const { return last_dim; }

    static utils::Result<UnweightedRMSNormInfo> create(
        infiniopTensorDescriptor_t y_desc,
        infiniopTensorDescriptor_t x_desc,
        float epsilon) {

        auto atype = y_desc->dtype();
        if (x_desc->dtype() != atype) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        if (atype != INFINI_DTYPE_F16 && atype != INFINI_DTYPE_BF16 &&
            atype != INFINI_DTYPE_F32 && atype != INFINI_DTYPE_F64) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        const size_t y_ndim = y_desc->ndim();
        const size_t x_ndim = x_desc->ndim();
        if (y_ndim != x_ndim || y_ndim < 2) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        for (size_t i = 0; i < y_ndim; ++i) {
            if (y_desc->dim(i) != x_desc->dim(i)) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
        }

        if (!is_contiguous(y_desc) || !is_contiguous(x_desc)) {
            return INFINI_STATUS_BAD_TENSOR_STRIDES;
        }

        const auto shape = y_desc->shape();
        const size_t last_dim = shape.back();
        if (last_dim == 0) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        size_t outer_size = 1;
        for (size_t i = 0; i + 1 < shape.size(); ++i) {
            outer_size *= shape[i];
        }

        return utils::Result<UnweightedRMSNormInfo>(UnweightedRMSNormInfo{
            atype,
            epsilon,
            shape,
            outer_size,
            last_dim,
        });
    }
};

} // namespace op::unweighted_rms_norm

#endif // __UNWEIGHTED_RMS_NORM_INFO_H__
