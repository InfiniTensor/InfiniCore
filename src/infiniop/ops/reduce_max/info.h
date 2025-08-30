#ifndef __REDUCE_MAX_INFO_H__
#define __REDUCE_MAX_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <vector>

namespace op::reduce_max {

class ReduceMaxInfo {
    ReduceMaxInfo() = default;

public:
    infiniDtype_t dtype;

    std::vector<size_t> shape;
    std::vector<ptrdiff_t> y_strides;
    std::vector<ptrdiff_t> x_strides;

    static utils::Result<ReduceMaxInfo> create(infiniopTensorDescriptor_t y_desc, infiniopTensorDescriptor_t x_desc, size_t dim) {
        auto dtype = y_desc->dtype();
        if (dtype != x_desc->dtype()) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32);

        size_t ndim = y_desc->ndim();
        if (x_desc->ndim() != ndim) {
            CHECK_STATUS(INFINI_STATUS_BAD_TENSOR_SHAPE);
        }
        CHECK_REDUCE_SHAPE(x_desc->shape(), dim, y_desc->shape());
        if (ndim > 4) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        } else if (ndim == 0) {
            std::vector<size_t> shape = {1, 1, 1, 1};
            std::vector<ptrdiff_t> y_strides = {0, 0, 0, 0};
            std::vector<ptrdiff_t> x_strides = {0, 0, 0, 0};
            return utils::Result<ReduceMaxInfo>(ReduceMaxInfo{
                dtype, shape, y_strides, x_strides});
        } else {
            std::vector<size_t> shape = x_desc->shape();
            std::vector<ptrdiff_t> y_strides = y_desc->strides();
            std::vector<ptrdiff_t> x_strides = x_desc->strides();
            if (dim != (shape.size() - 1)) {
                std::swap(shape[dim], shape[shape.size() - 1]);
                std::swap(y_strides[dim], y_strides[shape.size() - 1]);
                std::swap(x_strides[dim], x_strides[shape.size() - 1]);
            }
            while (shape.size() < 4) {
                shape.insert(shape.begin(), 1);
                y_strides.insert(y_strides.begin(), 0);
                x_strides.insert(x_strides.begin(), 0);
            }
            return utils::Result<ReduceMaxInfo>(ReduceMaxInfo{
                dtype, shape, y_strides, x_strides});
        }
    }
};

} // namespace op::reduce_max

#endif // __REDUCE_MAX_INFO_H__
