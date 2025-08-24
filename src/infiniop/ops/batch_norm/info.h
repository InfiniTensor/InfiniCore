#ifndef __BATCH_NORM_INFO_H__
#define __BATCH_NORM_INFO_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"

namespace op::batch_norm {

class BatchNormInfo {
private:
    BatchNormInfo() = default;

public:
//  ---------------------------- start: define member variables of Info ----------------------------
    infiniDtype_t dtype;
    size_t batch_size, channel_size, dim_size;

    ptrdiff_t running_mean_stride;
    ptrdiff_t running_var_stride;
    ptrdiff_t weight_stride;
    ptrdiff_t bias_stride;
    float momentum;
    float eps;

//  ----------------------------- end: define member variables of Info -----------------------------

    static utils::Result<BatchNormInfo> createBatchNormInfo(
        infiniopTensorDescriptor_t output_desc,
        infiniopTensorDescriptor_t running_mean_desc,
        infiniopTensorDescriptor_t running_var_desc,
        infiniopTensorDescriptor_t input_desc,
        infiniopTensorDescriptor_t weight_desc,
        infiniopTensorDescriptor_t bias_desc,
        float momentum,
        float eps
    ) {
//  ------------------------- start: check tensor shape and input validity -------------------------
        CHECK_OR_RETURN(
            input_desc->ndim() == 3, 
            INFINI_STATUS_BAD_TENSOR_SHAPE
        );        
        CHECK_SAME_SHAPE(output_desc->shape(), input_desc->shape());
        size_t batch_size = output_desc->dim(0),
            channel_size = output_desc->dim(1),
            dim_size = output_desc->dim(2);
        CHECK_SAME_SHAPE(
            running_mean_desc->shape(), running_var_desc->shape(),
            weight_desc->shape(), bias_desc->shape()
        );            
        CHECK_OR_RETURN(
            running_mean_desc->ndim() == 1 && running_mean_desc->dim(0) == channel_size,
            INFINI_STATUS_BAD_TENSOR_SHAPE
        );

//  -------------------------- end: check tensor shape and input validity --------------------------
        return utils::Result<BatchNormInfo>(BatchNormInfo{
//  ------------------------------ start: create an instance of Info -------------------------------
            output_desc->dtype(),
            batch_size, channel_size, dim_size,
            running_mean_desc->stride(0),
            running_var_desc->stride(0),
            weight_desc->stride(0),
            bias_desc->stride(0),
            momentum,
            eps
//  ------------------------------- end: create an instance of Info --------------------------------
        });
    }
};
}

#endif //  __BATCH_NORM_INFO_H__
