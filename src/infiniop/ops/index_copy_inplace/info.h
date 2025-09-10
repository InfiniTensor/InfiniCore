#ifndef __INDEX_COPY_INPLACE_INFO_H__
#define __INDEX_COPY_INPLACE_INFO_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"

namespace op::index_copy_inplace {

class IndexCopyInplaceInfo {
private:
    IndexCopyInplaceInfo() = default;

public:
//  ---------------------------- start: define member variables of Info ----------------------------
    infiniDtype_t dtype;
    size_t total_input_size;
	size_t total_output_size;
    std::vector<size_t> output_shape;
    std::vector<size_t> input_shape;
    std::vector<size_t> index_shape;
    std::vector<ptrdiff_t> output_strides;
    std::vector<ptrdiff_t> input_strides;
    std::vector<ptrdiff_t> index_strides;
    std::vector<ptrdiff_t> meta_strides;
    size_t dim;

//  ----------------------------- end: define member variables of Info -----------------------------

    static utils::Result<IndexCopyInplaceInfo> createIndexCopyInplaceInfo(
        infiniopTensorDescriptor_t output_desc,
        infiniopTensorDescriptor_t input_desc,
        infiniopTensorDescriptor_t index_desc,
        size_t dim
    ) {
//  ------------------------- start: check tensor shape and input validity -------------------------
        CHECK_OR_RETURN(output_desc->ndim() == input_desc->ndim(), INFINI_STATUS_BAD_TENSOR_STRIDES);
		std::vector<ptrdiff_t> meta_strides(input_desc->ndim());
		ptrdiff_t last_dim = 1;
		ptrdiff_t last_stride = 1;
		size_t total_input_size = 1;
		size_t total_output_size = 1;
		for (size_t d = 0; d < input_desc->ndim(); d++){
			total_input_size *= input_desc->dim(d);
			total_output_size *= output_desc->dim(d);
			if (d == dim) {
            	continue;
			}
			else {
				meta_strides[d] = last_dim * last_stride;
				last_dim = input_desc->dim(d);
				last_stride = meta_strides[d];
			}	
		}
		meta_strides[dim] = last_dim * last_stride;
//  -------------------------- end: check tensor shape and input validity --------------------------
        return utils::Result<IndexCopyInplaceInfo>(IndexCopyInplaceInfo{
//  ------------------------------ start: create an instance of Info -------------------------------
            output_desc->dtype(),
			total_input_size,
			total_output_size,
			output_desc->shape(),
			input_desc->shape(),
			index_desc->shape(),
			output_desc->strides(),
			input_desc->strides(),
			index_desc->strides(),
			meta_strides,
            dim
//  ------------------------------- end: create an instance of Info --------------------------------
        });
    }
};
}

#endif //  __INDEX_COPY_INPLACE_INFO_H__
