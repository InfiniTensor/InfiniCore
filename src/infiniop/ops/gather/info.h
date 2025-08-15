#ifndef __GATHER_INFO_H__
#define __GATHER_INFO_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"

namespace op::gather {

class GatherInfo {
private:
    GatherInfo() = default;

public:
//  ---------------------------- start: define member variables of Info ----------------------------
    infiniDtype_t dtype;
    size_t ndim;
    std::vector<size_t> output_shape;
    size_t input_dim_size;
    std::vector<ptrdiff_t> output_strides;
    std::vector<ptrdiff_t> input_strides;
    std::vector<ptrdiff_t> index_strides;
    size_t dim;

//  ----------------------------- end: define member variables of Info -----------------------------

    static utils::Result<GatherInfo> createGatherInfo(
        infiniopTensorDescriptor_t output_desc,
        infiniopTensorDescriptor_t input_desc,
        infiniopTensorDescriptor_t index_desc,
        size_t dim
    ) {
//  ------------------------- start: check tensor shape and input validity -------------------------
        CHECK_SAME_SHAPE(output_desc->shape(), index_desc->shape());
        size_t ndim = output_desc->ndim();
        for (size_t d = 0; d < ndim; d ++) {
            if (d != dim)
                CHECK_OR_RETURN(input_desc->dim(d) == output_desc->dim(d), INFINI_STATUS_BAD_TENSOR_SHAPE);
        }        
        CHECK_OR_RETURN(ndim > dim, INFINI_STATUS_BAD_PARAM);
//  -------------------------- end: check tensor shape and input validity --------------------------
        return utils::Result<GatherInfo>(GatherInfo{
//  ------------------------------ start: create an instance of Info -------------------------------
            output_desc->dtype(),
            ndim,
            output_desc->shape(),
            input_desc->dim(dim),
            output_desc->strides(),
            input_desc->strides(),
            index_desc->strides(),
            dim
//  ------------------------------- end: create an instance of Info --------------------------------
        });
    }
};
}

#endif //  __GATHER_INFO_H__
