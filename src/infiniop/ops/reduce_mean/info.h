#ifndef __REDUCE_MEAN_INFO_H__
#define __REDUCE_MEAN_INFO_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"

namespace op::reduce_mean {

class ReduceMeanInfo {
private:
    ReduceMeanInfo() = default;

public:
//  ---------------------------- start: define member variables of Info ----------------------------
    infiniDtype_t dtype;
	size_t ndim;
	std::vector<size_t> input_shape;
	std::vector<ptrdiff_t> output_strides;
	std::vector<ptrdiff_t> input_strides;
	size_t dim;

//  ----------------------------- end: define member variables of Info -----------------------------

    static utils::Result<ReduceMeanInfo> createReduceMeanInfo(
		infiniopTensorDescriptor_t output_desc,
		infiniopTensorDescriptor_t input_desc,
		size_t dim
    ) {
//  ------------------------- start: check tensor shape and input validity -------------------------
		size_t ndim = input_desc->ndim();
		CHECK_OR_RETURN(
			output_desc->ndim() == ndim,
			INFINI_STATUS_BAD_TENSOR_SHAPE
		);
		CHECK_OR_RETURN(
			ndim > dim,
			INFINI_STATUS_BAD_PARAM
		);
		for (size_t d = 0; d < ndim; d++) {
			CHECK_OR_RETURN(
				output_desc->dim(d) == ((d == dim) ? 1 : input_desc->dim(d)),
				INFINI_STATUS_BAD_TENSOR_SHAPE
			);
		}
//  -------------------------- end: check tensor shape and input validity --------------------------
        return utils::Result<ReduceMeanInfo>(ReduceMeanInfo{
//  ------------------------------ start: create an instance of Info -------------------------------
            output_desc->dtype(),
			ndim,
			input_desc->shape(),
			output_desc->strides(),
			input_desc->strides(),
			dim
//  ------------------------------- end: create an instance of Info --------------------------------
        });
    }
};
}

#endif //  __REDUCE_MEAN_INFO_H__
