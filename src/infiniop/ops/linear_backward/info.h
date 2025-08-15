#ifndef __LINEAR_BACKWARD_INFO_H__
#define __LINEAR_BACKWARD_INFO_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"

namespace op::linear_backward {

class LinearBackwardInfo {
private:
    LinearBackwardInfo() = default;

public:
//  ---------------------------- start: define member variables of Info ----------------------------
    infiniDtype_t dtype;
	size_t in_features;
	size_t out_features;
	ptrdiff_t grad_x_stride;
	ptrdiff_t grad_w_stride_out;
	ptrdiff_t grad_w_stride_in;
	ptrdiff_t grad_b_stride;
	ptrdiff_t grad_y_stride;
	ptrdiff_t x_stride;
	ptrdiff_t w_stride_out;
	ptrdiff_t w_stride_in;
	bool bias;

//  ----------------------------- end: define member variables of Info -----------------------------

    static utils::Result<LinearBackwardInfo> createLinearBackwardInfo(
		infiniopTensorDescriptor_t grad_x_desc,
		infiniopTensorDescriptor_t grad_w_desc,
		infiniopTensorDescriptor_t grad_b_desc,
		infiniopTensorDescriptor_t grad_y_desc,
		infiniopTensorDescriptor_t x_desc,
		infiniopTensorDescriptor_t w_desc
    ) {
//  ------------------------- start: check tensor shape and input validity -------------------------

		CHECK_SAME_SHAPE(x_desc->shape(), grad_x_desc->shape());
		CHECK_SAME_SHAPE(w_desc->shape(), grad_w_desc->shape());
		CHECK_OR_RETURN(
			w_desc->ndim() == 2 && x_desc->ndim() == 1 && grad_y_desc->ndim() == 1,
			INFINI_STATUS_BAD_TENSOR_SHAPE
		);
		size_t out = grad_y_desc->dim(0);
		size_t in = x_desc->dim(0);
		CHECK_OR_RETURN(
			w_desc->dim(0) == out && w_desc->dim(1) == in,
			INFINI_STATUS_BAD_TENSOR_SHAPE
		);		
        bool bias = (grad_b_desc != nullptr);
		if (bias)
			CHECK_OR_RETURN(
				grad_b_desc->ndim() == 1 && grad_b_desc->dim(0) == out,
				INFINI_STATUS_BAD_TENSOR_SHAPE   
			);	
		

//  -------------------------- end: check tensor shape and input validity --------------------------
        return utils::Result<LinearBackwardInfo>(LinearBackwardInfo{
//  ------------------------------ start: create an instance of Info -------------------------------
            grad_y_desc->dtype(),
			in,
			out,
			grad_x_desc->stride(0),
			grad_w_desc->stride(0),
			grad_w_desc->stride(1),
			bias ? (grad_b_desc->stride(0)) : 0,
			grad_y_desc->stride(0),
			x_desc->stride(0),
			w_desc->stride(0),
			w_desc->stride(1),
			bias
//  ------------------------------- end: create an instance of Info --------------------------------
        });
    }
};
}

#endif //  __LINEAR_BACKWARD_INFO_H__
