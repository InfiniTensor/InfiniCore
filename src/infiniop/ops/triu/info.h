#ifndef __TRIU_INFO_H__
#define __TRIU_INFO_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"

namespace op::triu {

class TriuInfo {
private:
    TriuInfo() = default;

public:
//  ---------------------------- start: define member variables of Info ----------------------------
    infiniDtype_t dtype;
	size_t elem_size;
	int row_size;
	int column_size;
	int diagonal;

//  ----------------------------- end: define member variables of Info -----------------------------

    static utils::Result<TriuInfo> createTriuInfo(
		infiniopTensorDescriptor_t output_desc,
		infiniopTensorDescriptor_t input_desc,
		int diagonal
    ) {
//  ------------------------- start: check tensor shape and input validity -------------------------
		CHECK_SAME_SHAPE(
			input_desc->shape(), output_desc->shape()
		);
		CHECK_OR_RETURN(
			input_desc->ndim() == 2,
			INFINI_STATUS_BAD_TENSOR_SHAPE
		);
		size_t elem_size = infiniSizeOf(input_desc->dtype());
//  -------------------------- end: check tensor shape and input validity --------------------------
        return utils::Result<TriuInfo>(TriuInfo{
//  ------------------------------ start: create an instance of Info -------------------------------
            input_desc->dtype(),
			elem_size,
			int(input_desc->dim(0)),
			int(input_desc->dim(1)),
			diagonal
//  ------------------------------- end: create an instance of Info --------------------------------
        });
    }
};
}

#endif //  __TRIU_INFO_H__
