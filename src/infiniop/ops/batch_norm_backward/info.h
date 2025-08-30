#ifndef __BATCH_NORM_BACKWARD_INFO_H__
#define __BATCH_NORM_BACKWARD_INFO_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"

namespace op::batch_norm_backward {

class BatchNormBackwardInfo {
private:
    BatchNormBackwardInfo() = default;

public:
//  ---------------------------- start: define member variables of Info ----------------------------
    infiniDtype_t dtype;
	size_t batch_size;
	size_t channel_size;
	size_t dim_size;
	ptrdiff_t grad_weight_stride;
	ptrdiff_t grad_bias_stride;
	ptrdiff_t weight_stride;
	ptrdiff_t running_mean_stride;
	ptrdiff_t running_var_stride;

//  ----------------------------- end: define member variables of Info -----------------------------

    static utils::Result<BatchNormBackwardInfo> createBatchNormBackwardInfo(
		infiniopTensorDescriptor_t grad_input_desc,
		infiniopTensorDescriptor_t grad_weight_desc,
		infiniopTensorDescriptor_t grad_bias_desc,
		infiniopTensorDescriptor_t input_desc,
		infiniopTensorDescriptor_t grad_output_desc,
		infiniopTensorDescriptor_t weight_desc,
		infiniopTensorDescriptor_t running_mean_desc,
		infiniopTensorDescriptor_t running_var_desc
    ) {
//  ------------------------- start: check tensor shape and input validity -------------------------
        CHECK_OR_RETURN(
            grad_input_desc->ndim() == 3, 
            INFINI_STATUS_BAD_TENSOR_SHAPE
        );        
        CHECK_SAME_SHAPE(grad_input_desc->shape(), grad_output_desc->shape(), input_desc->shape());
        size_t batch = grad_output_desc->dim(0),
            channel = grad_output_desc->dim(1),
            dim = grad_output_desc->dim(2);
        CHECK_SAME_SHAPE(
			grad_weight_desc->shape(), grad_bias_desc->shape(),weight_desc->shape(),
            running_mean_desc->shape(), running_var_desc->shape(),
        );            
        CHECK_OR_RETURN(
            grad_weight_desc->ndim() == 1 && grad_weight_desc->dim(0) == channel,
            INFINI_STATUS_BAD_TENSOR_SHAPE
        );
//  -------------------------- end: check tensor shape and input validity --------------------------
        return utils::Result<BatchNormBackwardInfo>(BatchNormBackwardInfo{
//  ------------------------------ start: create an instance of Info -------------------------------
            grad_output_desc->dtype(),
            batch, channel, dim,
			grad_weight_desc->stride(0),
			grad_bias_desc->stride(0),
            weight_desc->stride(0),
            running_mean_desc->stride(0),
            running_var_desc->stride(0)
//  ------------------------------- end: create an instance of Info --------------------------------
        });
    }
};
}

#endif //  __BATCH_NORM_BACKWARD_INFO_H__
