#ifndef __EQUAL_INFO_H__
#define __EQUAL_INFO_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"

namespace op::equal {

class EqualInfo {
private:
    EqualInfo() = default;

public:
//  ---------------------------- start: define member variables of Info ----------------------------
    size_t ndim;    
    infiniDtype_t dtype;
    std::vector<size_t> a_shape;
    std::vector<ptrdiff_t> a_strides;
    std::vector<ptrdiff_t> b_strides;

//  ----------------------------- end: define member variables of Info -----------------------------

    static utils::Result<EqualInfo> createEqualInfo(
        infiniopTensorDescriptor_t c_desc,
        infiniopTensorDescriptor_t a_desc,
        infiniopTensorDescriptor_t b_desc
    ) {
//  ------------------------- start: check tensor shape and input validity -------------------------
        CHECK_OR_RETURN(c_desc->ndim() == 1 && c_desc->dim(0) == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_SAME_SHAPE(a_desc->shape(), b_desc->shape());
//  -------------------------- end: check tensor shape and input validity --------------------------
        return utils::Result<EqualInfo>(EqualInfo{
//  ------------------------------ start: create an instance of Info -------------------------------
            a_desc->ndim(),            
            a_desc->dtype(),
            a_desc->shape(),
            a_desc->strides(),
            b_desc->strides()
//  ------------------------------- end: create an instance of Info --------------------------------
        });
    }
};
}

#endif //  __EQUAL_INFO_H__
