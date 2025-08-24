#ifndef __RMS_NORM_BACKWARD_INFO_H__
#define __RMS_NORM_BACKWARD_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <vector>

namespace op::rms_norm_backward {

class RMSNormBackwardInfo {
    RMSNormBackwardInfo() = default;

public:
    infiniDtype_t grad_x_dtype;
    infiniDtype_t grad_w_dtype;
    infiniDtype_t w_dtype;
    float epsilon;
    std::vector<size_t> shape;
    std::vector<ptrdiff_t> grad_x_strides;
    std::vector<ptrdiff_t> grad_w_strides;
    std::vector<ptrdiff_t> grad_y_strides;
    std::vector<ptrdiff_t> x_strides;
    std::vector<ptrdiff_t> w_strides;

    size_t ndim() const { return shape.size(); }
    size_t dim() const { return shape[ndim() - 1]; }
    size_t batch_size() const {
        size_t batch = 1;
        for (size_t i = 0; i < ndim() - 1; ++i) {
            batch *= shape[i];
        }
        return batch;
    }

    static utils::Result<RMSNormBackwardInfo> createRMSNormBackwardInfo(
        infiniopTensorDescriptor_t grad_x_desc,
        infiniopTensorDescriptor_t grad_w_desc,
        infiniopTensorDescriptor_t grad_y_desc,
        infiniopTensorDescriptor_t x_desc,
        infiniopTensorDescriptor_t w_desc,
        float epsilon) {

        auto grad_x_dtype = grad_x_desc->dtype();
        auto grad_w_dtype = grad_w_desc->dtype();
        auto w_dtype = w_desc->dtype();
        
        if (grad_y_desc->dtype() != grad_x_dtype || x_desc->dtype() != grad_x_dtype) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        
        if (grad_x_dtype == INFINI_DTYPE_F16 || grad_x_dtype == INFINI_DTYPE_BF16) {
            if (w_dtype != grad_x_dtype && w_dtype != INFINI_DTYPE_F32) {
                return INFINI_STATUS_BAD_TENSOR_DTYPE;
            }
            if (grad_w_dtype != grad_x_dtype && grad_w_dtype != INFINI_DTYPE_F32) {
                return INFINI_STATUS_BAD_TENSOR_DTYPE;
            }
        } else if (grad_x_dtype == INFINI_DTYPE_F32 || grad_x_dtype == INFINI_DTYPE_F64) {
            if (grad_x_dtype != w_dtype || grad_x_dtype != grad_w_dtype) {
                return INFINI_STATUS_BAD_TENSOR_DTYPE;
            }
        } else {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        if (grad_x_desc->ndim() < 2 || grad_y_desc->ndim() < 2 || 
            x_desc->ndim() < 2 || w_desc->ndim() != 1 || grad_w_desc->ndim() != 1) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        size_t dim = grad_x_desc->shape()[grad_x_desc->ndim() - 1];
        if (grad_y_desc->shape()[grad_y_desc->ndim() - 1] != dim ||
            x_desc->shape()[x_desc->ndim() - 1] != dim ||
            w_desc->shape()[0] != dim || grad_w_desc->shape()[0] != dim) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        RMSNormBackwardInfo info;
        info.grad_x_dtype = grad_x_dtype;
        info.grad_w_dtype = grad_w_dtype;
        info.w_dtype = w_dtype;
        info.epsilon = epsilon;
        info.shape = grad_x_desc->shape();
        info.grad_x_strides = grad_x_desc->strides();
        info.grad_w_strides = grad_w_desc->strides();
        info.grad_y_strides = grad_y_desc->strides();
        info.x_strides = x_desc->strides();
        info.w_strides = w_desc->strides();

        return utils::Result<RMSNormBackwardInfo>(info);
    }
};
}

#endif //  __RMS_NORM_BACKWARD_INFO_H__