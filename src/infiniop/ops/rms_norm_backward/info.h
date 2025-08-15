#ifndef __RMS_NORM_BACKWARD_INFO_H__
#define __RMS_NORM_BACKWARD_INFO_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"

namespace op::rms_norm_backward {

class RMSNormBackwardInfo {
private:
    RMSNormBackwardInfo() = default;

public:
//  ---------------------------- start: define member variables of Info ----------------------------
    infiniDtype_t dtype;
	size_t ndim;
	size_t total_size;
	size_t batch_size;
	size_t norm_size;
	std::vector<size_t> x_shape;
	std::vector<ptrdiff_t> grad_x_strides;
	std::vector<ptrdiff_t> grad_w_strides;
	std::vector<ptrdiff_t> grad_y_strides;
	std::vector<ptrdiff_t> x_strides;
	std::vector<ptrdiff_t> w_strides;
	
	inline size_t normalized_size() const {return x_shape[ndim - 1];}

//  ----------------------------- end: define member variables of Info -----------------------------

    static utils::Result<RMSNormBackwardInfo> createRMSNormBackwardInfo(
		infiniopTensorDescriptor_t grad_x_desc,
		infiniopTensorDescriptor_t grad_w_desc,
		infiniopTensorDescriptor_t grad_y_desc,
		infiniopTensorDescriptor_t x_desc,
		infiniopTensorDescriptor_t w_desc
    ) {
//  ------------------------- start: check tensor shape and input validity -------------------------
		size_t ndim = x_desc->ndim();
		size_t batch_size = 1, norm_size = x_desc->dim(ndim - 1);
		for (size_t d = 0; d < ndim - 1; d ++) {
			batch_size *= x_desc->dim(d);
		}
		CHECK_SAME_SHAPE(grad_x_desc->shape(), grad_y_desc->shape(), x_desc->shape());
		CHECK_SAME_SHAPE(grad_w_desc->shape(), w_desc->shape());
		CHECK_OR_RETURN(
			(w_desc->ndim() == 1) && (w_desc->dim(0) == norm_size),
			INFINI_STATUS_BAD_TENSOR_SHAPE
		);
//  -------------------------- end: check tensor shape and input validity --------------------------
        return utils::Result<RMSNormBackwardInfo>(RMSNormBackwardInfo{
//  ------------------------------ start: create an instance of Info -------------------------------
            grad_x_desc->dtype(),
			ndim,
			batch_size * norm_size,
			batch_size,
			norm_size,
			x_desc->shape(),
			grad_x_desc->strides(),
			grad_w_desc->strides(),
			grad_y_desc->strides(),
			x_desc->strides(),
			w_desc->strides()
//  ------------------------------- end: create an instance of Info --------------------------------
        });
    }
};
}

#endif //  __RMS_NORM_BACKWARD_INFO_H__
