#ifndef __SWIGLU_INFO_H__
#define __SWIGLU_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <vector>

namespace op::swiglu {

class SwiGLUInfo {
    SwiGLUInfo() = default;

public:
    infiniDtype_t dtype;
    std::vector<int64_t> shape;        // Output shape (half of combined input)
    std::vector<int64_t> combined_shape; // Combined input shape
    int64_t split_dim;                  // Dimension along which to split
    size_t numel;

    static utils::Result<SwiGLUInfo> create(
        infiniopTensorDescriptor_t output_desc,
        infiniopTensorDescriptor_t gate_desc,
        infiniopTensorDescriptor_t up_desc) {

        auto dtype = output_desc->dtype();

        // Check dtype compatibility
        CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);

        // Check shape compatibility
        auto output_shape_uint = output_desc->shape();
        auto gate_shape_uint = gate_desc->shape();
        auto up_shape_uint = up_desc->shape();

        auto ndim = output_desc->ndim();

        // Gate and up must have the same shape
        if (ndim != gate_desc->ndim() || ndim != up_desc->ndim()) {
            CHECK_STATUS(INFINI_STATUS_BAD_TENSOR_SHAPE);
        }

        // Convert to int64_t for comparison and storage
        std::vector<int64_t> output_shape(output_shape_uint.begin(), output_shape_uint.end());
        std::vector<int64_t> gate_shape(gate_shape_uint.begin(), gate_shape_uint.end());
        std::vector<int64_t> up_shape(up_shape_uint.begin(), up_shape_uint.end());
        
        CHECK_SAME_SHAPE(gate_shape, up_shape);
        
        // Output shape should match gate/up shape
        CHECK_SAME_SHAPE(output_shape, gate_shape);

        // Split along last dimension by default
        int64_t split_dim = static_cast<int64_t>(ndim) - 1;
        
        // Combined shape: double the split dimension
        std::vector<int64_t> combined_shape = gate_shape;
        combined_shape[split_dim] *= 2;

        size_t numel = 1;
        for (size_t i = 0; i < ndim; i++) {
            numel *= output_shape[i];
        }

        return utils::Result<SwiGLUInfo>(SwiGLUInfo{
            dtype,
            output_shape,
            combined_shape,
            split_dim,
            numel});
    }
};

} // namespace op::swiglu

#endif // __SWIGLU_INFO_H__
