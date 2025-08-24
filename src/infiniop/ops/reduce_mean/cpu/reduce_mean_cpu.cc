#include "reduce_mean_cpu.h"
#include "../../../reduce/cpu/reduce.h"
#include "../../../../utils.h"
#include "infinicore.h"

#ifdef ENABLE_OMP
#include <omp.h>
#endif

namespace op::reduce_mean::cpu {

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    size_t dim) {
    
    if (!handle || !desc_ptr || !output_desc || !input_desc) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    // Validate dimension
    if (dim >= input_desc->ndim()) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    // Validate data types
    if (input_desc->dtype() != output_desc->dtype()) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    
    infiniDtype_t dtype = input_desc->dtype();
    if (dtype != INFINI_DTYPE_F32 && dtype != INFINI_DTYPE_F16 && dtype != INFINI_DTYPE_BF16) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    
    // Validate shapes
    size_t ndim = input_desc->ndim();
    size_t output_ndim = output_desc->ndim();
    
    // Output should have same number of dimensions as input
    if (output_ndim != ndim) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }
    
    // Check that output shape matches input shape, with reduced dimension set to 1
    for (size_t i = 0; i < ndim; i++) {
        if (i == dim) {
            if (output_desc->shape()[i] != 1) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
        } else {
            if (output_desc->shape()[i] != input_desc->shape()[i]) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
        }
    }
    
    // Create info structure
    ReduceMeanInfo info;
    info.ndim = ndim;
    info.reduce_dim = dim;
    info.dtype = dtype;
    info.input_size = 1;
    info.output_size = 1;
    
    info.input_shape.resize(ndim);
    info.output_shape.resize(ndim);
    info.input_strides.resize(ndim);
    info.output_strides.resize(ndim);
    
    // Fill input shape and strides
    for (size_t i = 0; i < ndim; i++) {
        info.input_shape[i] = input_desc->shape()[i];
        info.input_strides[i] = input_desc->strides()[i];
        info.input_size *= input_desc->shape()[i];
    }
    
    // Fill output shape and strides
    for (size_t i = 0; i < ndim; i++) {
        if (i == dim) {
            info.output_shape[i] = 1;
        } else {
            info.output_shape[i] = input_desc->shape()[i];
        }
        info.output_strides[i] = output_desc->strides()[i];
        info.output_size *= info.output_shape[i];
    }
    
    *desc_ptr = new Descriptor(
        handle->device,
        handle->device_id,
        std::move(info),
        0
    );
    
    return INFINI_STATUS_SUCCESS;
}

template<typename T>
infiniStatus_t reduceMeanImpl(const ReduceMeanInfo &info, void *output, const void *input) {
    const T *input_ptr = static_cast<const T *>(input);
    T *output_ptr = static_cast<T *>(output);
    
    const size_t reduce_dim = info.reduce_dim;
    const size_t reduce_size = info.input_shape[reduce_dim];
    const ptrdiff_t reduce_stride = info.input_strides[reduce_dim];
    
    // Calculate the number of reduction operations
    size_t num_reductions = 1;
    for (size_t i = 0; i < info.ndim; i++) {
        if (i != reduce_dim) {
            num_reductions *= info.input_shape[i];
        }
    }
    
#ifdef ENABLE_OMP
#pragma omp parallel for
#endif
    for (size_t idx = 0; idx < num_reductions; idx++) {
        // Calculate the input and output offsets for this reduction
        size_t input_offset = 0;
        size_t output_offset = 0;
        size_t temp_idx = idx;
        
        for (size_t i = 0; i < info.ndim; i++) {
            if (i != reduce_dim) {
                size_t coord = temp_idx % info.input_shape[i];
                temp_idx /= info.input_shape[i];
                input_offset += coord * info.input_strides[i];
                output_offset += coord * info.output_strides[i];
            }
        }
        
        // Perform the reduction using the sum function
        T sum_result = op::common_cpu::reduce_op::sum(input_ptr + input_offset, reduce_size, reduce_stride);
        output_ptr[output_offset] = sum_result / static_cast<T>(reduce_size);
    }
    
    return INFINI_STATUS_SUCCESS;
}

template<typename T>
infiniStatus_t reduceMeanHalfImpl(const ReduceMeanInfo &info, void *output, const void *input) {
    static_assert(std::is_same<T, fp16_t>::value || std::is_same<T, bf16_t>::value,
                  "T must be fp16_t or bf16_t");
    
    const T *input_ptr = static_cast<const T *>(input);
    T *output_ptr = static_cast<T *>(output);
    
    const size_t reduce_dim = info.reduce_dim;
    const size_t reduce_size = info.input_shape[reduce_dim];
    const ptrdiff_t reduce_stride = info.input_strides[reduce_dim];
    
    // Calculate the number of reduction operations
    size_t num_reductions = 1;
    for (size_t i = 0; i < info.ndim; i++) {
        if (i != reduce_dim) {
            num_reductions *= info.input_shape[i];
        }
    }
    
#ifdef ENABLE_OMP
#pragma omp parallel for
#endif
    for (size_t idx = 0; idx < num_reductions; idx++) {
        // Calculate the input and output offsets for this reduction
        size_t input_offset = 0;
        size_t output_offset = 0;
        size_t temp_idx = idx;
        
        for (size_t i = 0; i < info.ndim; i++) {
            if (i != reduce_dim) {
                size_t coord = temp_idx % info.input_shape[i];
                temp_idx /= info.input_shape[i];
                input_offset += coord * info.input_strides[i];
                output_offset += coord * info.output_strides[i];
            }
        }
        
        // Perform the reduction using the common reduce function for half precision
        float sum_val = op::common_cpu::reduce_op::sum(
            input_ptr + input_offset,
            reduce_size,
            reduce_stride
        );
        
        // Calculate mean by dividing by reduce_size
        float mean_val = sum_val / static_cast<float>(reduce_size);
        
        output_ptr[output_offset] = utils::cast<T>(mean_val);
    }
    
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *output,
    const void *input,
    void *stream) const {
    
    switch (_info.dtype) {
        case INFINI_DTYPE_F32:
            return reduceMeanImpl<float>(_info, output, input);
        case INFINI_DTYPE_F16:
            return reduceMeanHalfImpl<fp16_t>(_info, output, input);
        case INFINI_DTYPE_BF16:
            return reduceMeanHalfImpl<bf16_t>(_info, output, input);
        default:
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::reduce_mean::cpu