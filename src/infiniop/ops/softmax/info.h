#ifndef __SOFTMAX_INFO_H__
#define __SOFTMAX_INFO_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"
#include <algorithm>

namespace op::softmax {
class SoftmaxInfo {
public:
    int axis;
    int otherdim_size;
    int stride;
    int size;
    int dimsize;

    static utils::Result<SoftmaxInfo> create(
        infiniopTensorDescriptor_t y_desc,
        infiniopTensorDescriptor_t x_desc,
        int axis) {

        if (y_desc->ndim() != x_desc->ndim()) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        SoftmaxInfo info;
        info.axis = axis;
        info.size = 1;
        info.otherdim_size = 1;
        info.stride = 1;
        info.dimsize = static_cast<int>(x_desc->dim(axis));
        int ndim = static_cast<int>(y_desc->ndim());
        for (int i = ndim - 1; i >= 0; i--) {
            info.size *= static_cast<int>(y_desc->dim(i));
        }
        info.stride = 1;
        for (int i = axis + 1; i < ndim; i++) {
            info.stride *= static_cast<int>(x_desc->dim(i));
        }
        info.otherdim_size = info.size / info.dimsize;
        return utils::Result<SoftmaxInfo>(info);
    }
};
} // namespace op::softmax

#endif // __SOFTMAX_INFO_H__
