#ifndef __SCATTER_INFO_H__
#define __SCATTER_INFO_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"

namespace op::scatter {

class ScatterInfo {
private:
    ScatterInfo() = default;

public:
//  ---------------------------- start: define member variables of Info ----------------------------
    infiniDtype_t dtype;
	size_t ndim;
	std::vector<size_t> output_shape;
	std::vector<size_t> input_shape;
	std::vector<size_t> index_shape;
	std::vector<ptrdiff_t> output_strides;
	std::vector<ptrdiff_t> input_strides;
	std::vector<ptrdiff_t> index_strides;
	size_t dim;

//  ----------------------------- end: define member variables of Info -----------------------------

    static utils::Result<ScatterInfo> createScatterInfo(
		infiniopTensorDescriptor_t output_desc,
		infiniopTensorDescriptor_t input_desc,
		infiniopTensorDescriptor_t index_desc,
		size_t dim
    ) {
//  ------------------------- start: check tensor shape and input validity -------------------------
		CHECK_OR_RETURN(
			input_desc->ndim() == output_desc->ndim() && output_desc->ndim() == index_desc->ndim(),
			INFINI_STATUS_BAD_TENSOR_SHAPE
		);
		size_t ndim = output_desc->ndim();
		for (size_t d = 0; d < ndim; d ++){
			if(d != dim) {
				CHECK_OR_RETURN(
					index_desc->dim(d) <= input_desc->dim(d) && index_desc->dim(d) <= output_desc->dim(d),
					INFINI_STATUS_BAD_TENSOR_SHAPE;
				);
			}
		}
		CHECK_OR_RETURN(index_desc->dim(dim) <= input_desc->dim(dim), INFINI_STATUS_BAD_TENSOR_SHAPE);		
//  -------------------------- end: check tensor shape and input validity --------------------------
        return utils::Result<ScatterInfo>(ScatterInfo{
//  ------------------------------ start: create an instance of Info -------------------------------
            output_desc->dtype(),
			ndim,
			output_desc->shape(),
			input_desc->shape(),
			index_desc->shape(),
			output_desc->strides(),
			input_desc->strides(),
			index_desc->strides(),
			dim
//  ------------------------------- end: create an instance of Info --------------------------------
        });
    }
};
}

#endif //  __SCATTER_INFO_H__
