#ifndef __ADD_INFO_H__
#define __ADD_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <vector>

namespace op::add {

class AddInfo {
    AddInfo() = default;

public:
    infiniDtype_t dtype;
    std::vector<int64_t> shape;
    size_t numel;

    static utils::Result<AddInfo> create(
        infiniopTensorDescriptor_t c_desc,
        infiniopTensorDescriptor_t a_desc,
        infiniopTensorDescriptor_t b_desc) {
        
        auto dtype = c_desc->dtype();
        
        // Check dtype compatibility
        CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);
        
        // Check shape compatibility (broadcast)
        auto c_shape = c_desc->shape();
        auto a_shape = a_desc->shape();
        auto b_shape = b_desc->shape();
        
        auto c_ndim = c_desc->ndim();
        
        // Require same ndim and shape for now
        if (c_ndim != a_desc->ndim() || c_ndim != b_desc->ndim()) {
            CHECK_STATUS(INFINI_STATUS_BAD_TENSOR_SHAPE);
        }
        
        CHECK_SAME_SHAPE(c_shape, a_shape);
        CHECK_SAME_SHAPE(c_shape, b_shape);
        
        size_t numel = 1;
        std::vector<int64_t> shape;
        for (size_t i = 0; i < c_ndim; i++) {
            shape.push_back(c_shape[i]);
            numel *= c_shape[i];
        }
        
        return utils::Result<AddInfo>(AddInfo{
            dtype,
            shape,
            numel});
    }
};

} // namespace op::add

#endif // __ADD_INFO_H__
