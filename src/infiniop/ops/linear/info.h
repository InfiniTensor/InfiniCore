#ifndef __LINEAR_INFO_H__
#define __LINEAR_INFO_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"

namespace op::linear {

class LinearInfo {
private:
    LinearInfo() = default;

public:
//  ---------------------------- start: define member variables of Info ----------------------------
    infiniDtype_t dtype;
	size_t in_features;
	size_t out_features;
	ptrdiff_t y_stride;
	ptrdiff_t x_stride;
	ptrdiff_t w_stride_in;
    ptrdiff_t w_stride_out;
	ptrdiff_t b_stride;
    bool bias;

//  ----------------------------- end: define member variables of Info -----------------------------

    static utils::Result<LinearInfo> createLinearInfo(
        infiniopTensorDescriptor_t y_desc,
        infiniopTensorDescriptor_t x_desc,
        infiniopTensorDescriptor_t w_desc,
        infiniopTensorDescriptor_t b_desc
    ) {
//  ------------------------- start: check tensor shape and input validity -------------------------
        size_t in_features = x_desc->dim(0);
        size_t out_features = y_desc->dim(0);
        CHECK_OR_RETURN(x_desc->ndim() == 1 && y_desc->ndim() == 1 && w_desc->ndim() == 2 && \
            w_desc->dim(0) == out_features && w_desc->dim(1) == in_features,
            INFINI_STATUS_BAD_TENSOR_SHAPE    
        );
        bool bias = (b_desc != nullptr);
        if (bias)
            CHECK_OR_RETURN(
                b_desc->ndim() == 1 && b_desc->dim(0) == out_features,
                INFINI_STATUS_BAD_TENSOR_SHAPE   
            );
        

//  -------------------------- end: check tensor shape and input validity --------------------------
        return utils::Result<LinearInfo>(LinearInfo{
//  ------------------------------ start: create an instance of Info -------------------------------
            y_desc->dtype(),
			x_desc->dim(0),
            y_desc->dim(0),
			y_desc->stride(0),
            x_desc->stride(0),
            w_desc->stride(1),
            w_desc->stride(0),
            bias ? b_desc->stride(0) : 0,
            bias
//  ------------------------------- end: create an instance of Info --------------------------------
        });
    }
};
}

#endif //  __LINEAR_INFO_H__
