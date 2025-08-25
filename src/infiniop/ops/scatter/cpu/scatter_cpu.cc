#include "scatter_cpu.h"
#include "../../../../utils.h"
#include "../../../tensor.h"
#include <cstring>
#include <omp.h>

namespace op::scatter::cpu {

template<typename T, typename IndexT>
static infiniStatus_t calculate_scatter(
    const std::vector<size_t> &input_shape,
    const std::vector<size_t> &output_shape,
    const std::vector<size_t> &src_shape,
    const std::vector<ptrdiff_t> &input_strides,
    const std::vector<ptrdiff_t> &output_strides,
    const std::vector<ptrdiff_t> &index_strides,
    const std::vector<ptrdiff_t> &src_strides,
    size_t dim,
    T * output,
    const T * input,
    const IndexT * index,
    const T * src
) {
    // Step 1: Copy input to output (initialization)
    size_t total_input_elements = 1;
    for (size_t d = 0; d < input_shape.size(); d++) {
        total_input_elements *= input_shape[d];
    }
    
    std::memcpy(output, input, total_input_elements * sizeof(T));
    
    // Step 2: Scatter operation - iterate over src elements
    size_t total_src_elements = 1;
    for (size_t d = 0; d < src_shape.size(); d++) {
        total_src_elements *= src_shape[d];
    }
    
    for(size_t linear_idx = 0; linear_idx < total_src_elements; linear_idx++) {
        
        // Convert linear index to multi-dimensional coordinates
        std::vector<size_t> coords(src_shape.size());
        size_t temp_idx = linear_idx;
        for (int d = src_shape.size() - 1; d >= 0; d--) {
            coords[d] = temp_idx % src_shape[d];
            temp_idx /= src_shape[d];
        }
        
        // Calculate src offset
        size_t src_offset = 0;
        for (size_t d = 0; d < src_shape.size(); d++) {
            src_offset += coords[d] * src_strides[d];
        }
        
        // Calculate index offset
        size_t index_offset = 0;
        for (size_t d = 0; d < src_shape.size(); d++) {
            index_offset += coords[d] * index_strides[d];
        }
        
        // Get scatter index and perform bounds check
        IndexT scatter_index = *((const IndexT*)((const char*)index + index_offset));
        if (scatter_index < 0 || scatter_index >= static_cast<IndexT>(output_shape[dim])) {
            continue; // Skip invalid indices
        }
        
        // Calculate output offset - for scatter operation, use scatter_index in dim dimension
        size_t output_offset = 0;
        for (size_t d = 0; d < output_shape.size(); d++) {
            if (d == dim) {
                output_offset += scatter_index * output_strides[d];
            } else {
                output_offset += coords[d] * output_strides[d];
            }
        }
        
        // Copy data from src to output
        std::memcpy((char*)output + output_offset, (const char*)src + src_offset, sizeof(T));
    }
    
    return INFINI_STATUS_SUCCESS;
}

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t index_desc,
    infiniopTensorDescriptor_t src_desc,
    int dim) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto dtype = input_desc->dtype();

    // Check data types - support all legal types
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64, INFINI_DTYPE_BF16,
                INFINI_DTYPE_I8, INFINI_DTYPE_I16, INFINI_DTYPE_I32, INFINI_DTYPE_I64,
                INFINI_DTYPE_U8, INFINI_DTYPE_U16, INFINI_DTYPE_U32, INFINI_DTYPE_U64,
                INFINI_DTYPE_BOOL);

    // Check that input, output and src have same dtype
    if (input_desc->dtype() != output_desc->dtype() || input_desc->dtype() != src_desc->dtype()) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    // Check that index is integer type
    auto index_dtype = index_desc->dtype();
    if (index_dtype != INFINI_DTYPE_I32 && index_dtype != INFINI_DTYPE_I64) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    // Validate dimensions
    auto input_shape = input_desc->shape();
    auto output_shape = output_desc->shape();
    auto src_shape = src_desc->shape();
    auto index_shape = index_desc->shape();
    
    if (dim < 0 || dim >= static_cast<int>(input_shape.size())) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    // Check that input and output have same shape
    if (input_shape != output_shape) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    // Check that src and index have same shape
    if (src_shape != index_shape) {
        return INFINI_STATUS_BAD_PARAM;
    }

    auto desc = new Descriptor();
    desc->_input_desc = input_desc;
    desc->_output_desc = output_desc;
    desc->_index_desc = index_desc;
    desc->_src_desc = src_desc;
    desc->_dim = dim;
    desc->_input_shape = input_shape;
    desc->_output_shape = output_shape;
    desc->_src_shape = src_shape;
    desc->_input_strides = input_desc->getByteStrides();
    desc->_output_strides = output_desc->getByteStrides();
    desc->_index_strides = index_desc->getByteStrides();
    desc->_src_strides = src_desc->getByteStrides();
    desc->_dtype = dtype;
    desc->_index_dtype = index_dtype;
    desc->_handle = handle;
    desc->device_type = handle_->device;
    desc->device_id = handle_->device_id;

    *desc_ptr = desc;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    const void *index,
    const void *src,
    void *stream) const {

    auto input_shape = _input_shape;
    auto output_shape = _output_shape;
    auto src_shape = _src_shape;
    auto input_strides = _input_strides;
    auto output_strides = _output_strides;
    auto index_strides = _index_strides;
    auto src_strides = _src_strides;
    auto dtype = _dtype;
    auto index_dtype = _index_dtype;
    
    // Call kernel based on data type and index type
    if (index_dtype == INFINI_DTYPE_I32) {
        switch (dtype) {
            case INFINI_DTYPE_F16:
                return calculate_scatter<fp16_t, int32_t>(
                     input_shape, output_shape, src_shape,
                     input_strides, output_strides, index_strides, src_strides,
                     static_cast<size_t>(_dim),
                     static_cast<fp16_t*>(output),
                     static_cast<const fp16_t*>(input),
                     static_cast<const int32_t*>(index),
                     static_cast<const fp16_t*>(src));
                break;
            case INFINI_DTYPE_F32:
                return calculate_scatter<float, int32_t>(
                     input_shape, output_shape, src_shape,
                     input_strides, output_strides, index_strides, src_strides,
                     static_cast<size_t>(_dim),
                     static_cast<float*>(output),
                     static_cast<const float*>(input),
                     static_cast<const int32_t*>(index),
                     static_cast<const float*>(src));
                 break;
             case INFINI_DTYPE_F64:
                 return calculate_scatter<double, int32_t>(
                      input_shape, output_shape, src_shape,
                      input_strides, output_strides, index_strides, src_strides,
                      static_cast<size_t>(_dim),
                      static_cast<double*>(output),
                      static_cast<const double*>(input),
                      static_cast<const int32_t*>(index),
                      static_cast<const double*>(src));
                 break;
             case INFINI_DTYPE_BF16:
                 return calculate_scatter<bf16_t, int32_t>(
                      input_shape, output_shape, src_shape,
                      input_strides, output_strides, index_strides, src_strides,
                      static_cast<size_t>(_dim),
                      static_cast<bf16_t*>(output),
                      static_cast<const bf16_t*>(input),
                      static_cast<const int32_t*>(index),
                      static_cast<const bf16_t*>(src));
                 break;
             case INFINI_DTYPE_I8:
                 return calculate_scatter<int8_t, int32_t>(
                      input_shape, output_shape, src_shape,
                      input_strides, output_strides, index_strides, src_strides,
                      static_cast<size_t>(_dim),
                      static_cast<int8_t*>(output),
                      static_cast<const int8_t*>(input),
                      static_cast<const int32_t*>(index),
                      static_cast<const int8_t*>(src));
                 break;
             case INFINI_DTYPE_I16:
                 return calculate_scatter<int16_t, int32_t>(
                      input_shape, output_shape, src_shape,
                      input_strides, output_strides, index_strides, src_strides,
                      static_cast<size_t>(_dim),
                      static_cast<int16_t*>(output),
                      static_cast<const int16_t*>(input),
                      static_cast<const int32_t*>(index),
                      static_cast<const int16_t*>(src));
                 break;
             case INFINI_DTYPE_I32:
                 return calculate_scatter<int32_t, int32_t>(
                      input_shape, output_shape, src_shape,
                      input_strides, output_strides, index_strides, src_strides,
                      static_cast<size_t>(_dim),
                      static_cast<int32_t*>(output),
                      static_cast<const int32_t*>(input),
                      static_cast<const int32_t*>(index),
                      static_cast<const int32_t*>(src));
                 break;
             case INFINI_DTYPE_I64:
                 return calculate_scatter<int64_t, int32_t>(
                      input_shape, output_shape, src_shape,
                      input_strides, output_strides, index_strides, src_strides,
                      static_cast<size_t>(_dim),
                      static_cast<int64_t*>(output),
                      static_cast<const int64_t*>(input),
                      static_cast<const int32_t*>(index),
                      static_cast<const int64_t*>(src));
                 break;
             case INFINI_DTYPE_U8:
                 return calculate_scatter<uint8_t, int32_t>(
                      input_shape, output_shape, src_shape,
                      input_strides, output_strides, index_strides, src_strides,
                      static_cast<size_t>(_dim),
                      static_cast<uint8_t*>(output),
                      static_cast<const uint8_t*>(input),
                      static_cast<const int32_t*>(index),
                      static_cast<const uint8_t*>(src));
                 break;
             case INFINI_DTYPE_U16:
                 return calculate_scatter<uint16_t, int32_t>(
                      input_shape, output_shape, src_shape,
                      input_strides, output_strides, index_strides, src_strides,
                      static_cast<size_t>(_dim),
                      static_cast<uint16_t*>(output),
                      static_cast<const uint16_t*>(input),
                      static_cast<const int32_t*>(index),
                      static_cast<const uint16_t*>(src));
                 break;
             case INFINI_DTYPE_U32:
                 return calculate_scatter<uint32_t, int32_t>(
                      input_shape, output_shape, src_shape,
                      input_strides, output_strides, index_strides, src_strides,
                      static_cast<size_t>(_dim),
                      static_cast<uint32_t*>(output),
                      static_cast<const uint32_t*>(input),
                      static_cast<const int32_t*>(index),
                      static_cast<const uint32_t*>(src));
                 break;
             case INFINI_DTYPE_U64:
                 return calculate_scatter<uint64_t, int32_t>(
                      input_shape, output_shape, src_shape,
                      input_strides, output_strides, index_strides, src_strides,
                      static_cast<size_t>(_dim),
                      static_cast<uint64_t*>(output),
                      static_cast<const uint64_t*>(input),
                      static_cast<const int32_t*>(index),
                      static_cast<const uint64_t*>(src));
                 break;
             case INFINI_DTYPE_BOOL:
                 return calculate_scatter<bool, int32_t>(
                      input_shape, output_shape, src_shape,
                      input_strides, output_strides, index_strides, src_strides,
                      static_cast<size_t>(_dim),
                      static_cast<bool*>(output),
                      static_cast<const bool*>(input),
                      static_cast<const int32_t*>(index),
                      static_cast<const bool*>(src));
                 break;
             default:
                 return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
    } else if (index_dtype == INFINI_DTYPE_I64) {
        switch (dtype) {
            case INFINI_DTYPE_F16:
                return calculate_scatter<fp16_t, int64_t>(
                     input_shape, output_shape, src_shape,
                     input_strides, output_strides, index_strides, src_strides,
                     static_cast<size_t>(_dim),
                     static_cast<fp16_t*>(output),
                     static_cast<const fp16_t*>(input),
                     static_cast<const int64_t*>(index),
                     static_cast<const fp16_t*>(src));
                break;
            case INFINI_DTYPE_F32:
                return calculate_scatter<float, int64_t>(
                     input_shape, output_shape, src_shape,
                     input_strides, output_strides, index_strides, src_strides,
                     static_cast<size_t>(_dim),
                     static_cast<float*>(output),
                     static_cast<const float*>(input),
                     static_cast<const int64_t*>(index),
                     static_cast<const float*>(src));
                 break;
             case INFINI_DTYPE_F64:
                 return calculate_scatter<double, int64_t>(
                      input_shape, output_shape, src_shape,
                      input_strides, output_strides, index_strides, src_strides,
                      static_cast<size_t>(_dim),
                      static_cast<double*>(output),
                      static_cast<const double*>(input),
                      static_cast<const int64_t*>(index),
                      static_cast<const double*>(src));
                 break;
             case INFINI_DTYPE_BF16:
                 return calculate_scatter<bf16_t, int64_t>(
                      input_shape, output_shape, src_shape,
                      input_strides, output_strides, index_strides, src_strides,
                      static_cast<size_t>(_dim),
                      static_cast<bf16_t*>(output),
                      static_cast<const bf16_t*>(input),
                      static_cast<const int64_t*>(index),
                      static_cast<const bf16_t*>(src));
                 break;
             case INFINI_DTYPE_I8:
                 return calculate_scatter<int8_t, int64_t>(
                      input_shape, output_shape, src_shape,
                      input_strides, output_strides, index_strides, src_strides,
                      static_cast<size_t>(_dim),
                      static_cast<int8_t*>(output),
                      static_cast<const int8_t*>(input),
                      static_cast<const int64_t*>(index),
                      static_cast<const int8_t*>(src));
                 break;
             case INFINI_DTYPE_I16:
                 return calculate_scatter<int16_t, int64_t>(
                      input_shape, output_shape, src_shape,
                      input_strides, output_strides, index_strides, src_strides,
                      static_cast<size_t>(_dim),
                      static_cast<int16_t*>(output),
                      static_cast<const int16_t*>(input),
                      static_cast<const int64_t*>(index),
                      static_cast<const int16_t*>(src));
                 break;
             case INFINI_DTYPE_I32:
                 return calculate_scatter<int32_t, int64_t>(
                      input_shape, output_shape, src_shape,
                      input_strides, output_strides, index_strides, src_strides,
                      static_cast<size_t>(_dim),
                      static_cast<int32_t*>(output),
                      static_cast<const int32_t*>(input),
                      static_cast<const int64_t*>(index),
                      static_cast<const int32_t*>(src));
                 break;
             case INFINI_DTYPE_I64:
                 return calculate_scatter<int64_t, int64_t>(
                      input_shape, output_shape, src_shape,
                      input_strides, output_strides, index_strides, src_strides,
                      static_cast<size_t>(_dim),
                      static_cast<int64_t*>(output),
                      static_cast<const int64_t*>(input),
                      static_cast<const int64_t*>(index),
                      static_cast<const int64_t*>(src));
                 break;
             case INFINI_DTYPE_U8:
                 return calculate_scatter<uint8_t, int64_t>(
                      input_shape, output_shape, src_shape,
                      input_strides, output_strides, index_strides, src_strides,
                      static_cast<size_t>(_dim),
                      static_cast<uint8_t*>(output),
                      static_cast<const uint8_t*>(input),
                      static_cast<const int64_t*>(index),
                      static_cast<const uint8_t*>(src));
                 break;
             case INFINI_DTYPE_U16:
                 return calculate_scatter<uint16_t, int64_t>(
                      input_shape, output_shape, src_shape,
                      input_strides, output_strides, index_strides, src_strides,
                      static_cast<size_t>(_dim),
                      static_cast<uint16_t*>(output),
                      static_cast<const uint16_t*>(input),
                      static_cast<const int64_t*>(index),
                      static_cast<const uint16_t*>(src));
                 break;
             case INFINI_DTYPE_U32:
                 return calculate_scatter<uint32_t, int64_t>(
                      input_shape, output_shape, src_shape,
                      input_strides, output_strides, index_strides, src_strides,
                      static_cast<size_t>(_dim),
                      static_cast<uint32_t*>(output),
                      static_cast<const uint32_t*>(input),
                      static_cast<const int64_t*>(index),
                      static_cast<const uint32_t*>(src));
                 break;
             case INFINI_DTYPE_U64:
                 return calculate_scatter<uint64_t, int64_t>(
                      input_shape, output_shape, src_shape,
                      input_strides, output_strides, index_strides, src_strides,
                      static_cast<size_t>(_dim),
                      static_cast<uint64_t*>(output),
                      static_cast<const uint64_t*>(input),
                      static_cast<const int64_t*>(index),
                      static_cast<const uint64_t*>(src));
                 break;
             case INFINI_DTYPE_BOOL:
                 return calculate_scatter<bool, int64_t>(
                      input_shape, output_shape, src_shape,
                      input_strides, output_strides, index_strides, src_strides,
                      static_cast<size_t>(_dim),
                      static_cast<bool*>(output),
                      static_cast<const bool*>(input),
                      static_cast<const int64_t*>(index),
                      static_cast<const bool*>(src));
                 break;
             default:
                 return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::scatter::cpu