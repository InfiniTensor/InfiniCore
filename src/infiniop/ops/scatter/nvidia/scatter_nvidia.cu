#include "scatter_nvidia.h"
#include "../cuda/kernel.cuh"
#include "../../../../utils.h"
#include "../../../tensor.h"
#include <cuda_runtime.h>
#include <cstring>

namespace op::scatter::nvidia {

// Helper function to calculate scatter using CUDA kernel
template<typename T, typename IndexT>
static infiniStatus_t calculate_scatter_cuda(
    const std::vector<size_t> &input_shape,
    const std::vector<size_t> &output_shape,
    const std::vector<size_t> &src_shape,
    const std::vector<size_t> &index_shape,
    const std::vector<ptrdiff_t> &input_strides,
    const std::vector<ptrdiff_t> &output_strides,
    const std::vector<ptrdiff_t> &index_strides,
    const std::vector<ptrdiff_t> &src_strides,
    size_t dim,
    T * output,
    const T * input,
    const IndexT * index,
    const T * src,
    cudaStream_t stream
) {
    // Scatter operation - iterate over src elements
    // Note: Input to output initialization is handled by the test framework
    size_t total_src_elements = 1;
    for (size_t d = 0; d < src_shape.size(); d++) {
        total_src_elements *= src_shape[d];
    }
    
    if (total_src_elements == 0) {
        return INFINI_STATUS_SUCCESS;
    }
    
    // Allocate device memory for shape and stride arrays
    size_t* d_src_shape = nullptr;
    size_t* d_output_shape = nullptr;
    size_t* d_src_strides = nullptr;
    size_t* d_output_strides = nullptr;
    size_t* d_index_strides = nullptr;

    
    // Allocate device memory for shapes and strides
    cudaError_t err = cudaMalloc(&d_src_shape, src_shape.size() * sizeof(size_t));
    if (err != cudaSuccess) {
        return INFINI_STATUS_INTERNAL_ERROR;
    }
    
    err = cudaMalloc(&d_output_shape, output_shape.size() * sizeof(size_t));
    if (err != cudaSuccess) {
        cudaFree(d_src_shape);
        return INFINI_STATUS_INTERNAL_ERROR;
    }
    
    err = cudaMalloc(&d_src_strides, src_strides.size() * sizeof(size_t));
    if (err != cudaSuccess) {
        cudaFree(d_src_shape);
        cudaFree(d_output_shape);
        return INFINI_STATUS_INTERNAL_ERROR;
    }
    
    err = cudaMalloc(&d_output_strides, output_strides.size() * sizeof(size_t));
    if (err != cudaSuccess) {
        cudaFree(d_src_shape);
        cudaFree(d_output_shape);
        cudaFree(d_src_strides);
        return INFINI_STATUS_INTERNAL_ERROR;
    }
    
    err = cudaMalloc(&d_index_strides, index_strides.size() * sizeof(size_t));
    if (err != cudaSuccess) {
        cudaFree(d_src_shape);
        cudaFree(d_output_shape);
        cudaFree(d_src_strides);
        cudaFree(d_output_strides);
        return INFINI_STATUS_INTERNAL_ERROR;
    }
    

    
    // Convert byte strides to element strides for kernel
    std::vector<size_t> h_src_strides(src_strides.size());
    std::vector<size_t> h_output_strides(output_strides.size());
    std::vector<size_t> h_index_strides(index_strides.size());
    
    // Convert byte strides to element strides
    for (size_t i = 0; i < src_strides.size(); i++) {
        h_src_strides[i] = static_cast<size_t>(src_strides[i]) / sizeof(T);
    }
    for (size_t i = 0; i < index_strides.size(); i++) {
        h_index_strides[i] = static_cast<size_t>(index_strides[i]) / sizeof(IndexT);
    }
    for (size_t i = 0; i < output_strides.size(); i++) {
        h_output_strides[i] = static_cast<size_t>(output_strides[i]) / sizeof(T);
    }
    
    // Copy shape and stride data to device
    err = cudaMemcpy(d_src_shape, src_shape.data(), src_shape.size() * sizeof(size_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_src_shape);
        cudaFree(d_output_shape);
        cudaFree(d_src_strides);
        cudaFree(d_output_strides);
        cudaFree(d_index_strides);
        return INFINI_STATUS_INTERNAL_ERROR;
    }
    
    err = cudaMemcpy(d_output_shape, output_shape.data(), output_shape.size() * sizeof(size_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_src_shape);
        cudaFree(d_output_shape);
        cudaFree(d_src_strides);
        cudaFree(d_output_strides);
        cudaFree(d_index_strides);
        return INFINI_STATUS_INTERNAL_ERROR;
    }
    
    err = cudaMemcpy(d_src_strides, h_src_strides.data(), h_src_strides.size() * sizeof(size_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_src_shape);
        cudaFree(d_output_shape);
        cudaFree(d_src_strides);
        cudaFree(d_output_strides);
        cudaFree(d_index_strides);
        return INFINI_STATUS_INTERNAL_ERROR;
    }
    
    err = cudaMemcpy(d_output_strides, h_output_strides.data(), h_output_strides.size() * sizeof(size_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_src_shape);
        cudaFree(d_output_shape);
        cudaFree(d_src_strides);
        cudaFree(d_output_strides);
        cudaFree(d_index_strides);
        return INFINI_STATUS_INTERNAL_ERROR;
    }
    
    err = cudaMemcpy(d_index_strides, h_index_strides.data(), h_index_strides.size() * sizeof(size_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_src_shape);
        cudaFree(d_output_shape);
        cudaFree(d_src_strides);
        cudaFree(d_output_strides);
        cudaFree(d_index_strides);
        return INFINI_STATUS_INTERNAL_ERROR;
    }
    
    // Calculate index size
    size_t total_index_elements = 1;
    for (size_t d = 0; d < index_shape.size(); d++) {
        total_index_elements *= index_shape[d];
    }
    
    // Calculate input size
    size_t total_input_elements = 1;
    for (size_t d = 0; d < input_shape.size(); d++) {
        total_input_elements *= input_shape[d];
    }
    
    // Calculate output size
    size_t total_output_elements = 1;
    for (size_t d = 0; d < output_shape.size(); d++) {
        total_output_elements *= output_shape[d];
    }
    
    // Step 1: Copy input to output (initialization step)
    size_t input_size_bytes = total_input_elements * sizeof(T);
    
    // Check if pointers are valid
    if (input == nullptr) {
        return INFINI_STATUS_BAD_PARAM;
    }
    if (output == nullptr) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    err = cudaMemcpy(output, input, input_size_bytes, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_src_shape);
        cudaFree(d_output_shape);
        cudaFree(d_src_strides);
        cudaFree(d_output_strides);
        cudaFree(d_index_strides);

        return INFINI_STATUS_INTERNAL_ERROR;
    }
    
    // Step 2: Launch scatter kernel using CUDA kernel from cuda/kernel.cuh
    
    // Synchronize to ensure all previous operations are complete
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(d_src_shape);
        cudaFree(d_output_shape);
        cudaFree(d_src_strides);
        cudaFree(d_output_strides);
        cudaFree(d_index_strides);
        return INFINI_STATUS_INTERNAL_ERROR;
    }
    
    err = op::scatter::cuda::launch_scatter_kernel<T, IndexT>(
        output, src, index, total_src_elements, total_index_elements, total_output_elements, dim,
        d_src_strides, d_output_strides, d_index_strides,
        d_src_shape, d_output_shape, src_shape.size()
    );
    
    if (err != cudaSuccess) {
        cudaFree(d_src_shape);
        cudaFree(d_output_shape);
        cudaFree(d_src_strides);
        cudaFree(d_output_strides);
        cudaFree(d_index_strides);
    
        return INFINI_STATUS_INTERNAL_ERROR;
    }
    
    // Clean up device memory
    cudaFree(d_src_shape);
    cudaFree(d_output_shape);
    cudaFree(d_src_strides);
    cudaFree(d_output_strides);
    cudaFree(d_index_strides);
    
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

    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    auto dtype = input_desc->dtype();

    // Check that input data type is supported
    if (dtype != INFINI_DTYPE_F16 && dtype != INFINI_DTYPE_F32 && 
        dtype != INFINI_DTYPE_F64 && dtype != INFINI_DTYPE_BF16 &&
        dtype != INFINI_DTYPE_I8 && dtype != INFINI_DTYPE_I16 && 
        dtype != INFINI_DTYPE_I32 && dtype != INFINI_DTYPE_I64 &&
        dtype != INFINI_DTYPE_U8 && dtype != INFINI_DTYPE_U16 && 
        dtype != INFINI_DTYPE_U32 && dtype != INFINI_DTYPE_U64 &&
        dtype != INFINI_DTYPE_BOOL) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

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
    desc->_index_shape = index_shape;
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
    auto index_shape = _index_shape;
    auto input_strides = _input_strides;
    auto output_strides = _output_strides;
    auto index_strides = _index_strides;
    auto src_strides = _src_strides;
    auto dtype = _dtype;
    auto index_dtype = _index_dtype;
    auto cuda_stream = static_cast<cudaStream_t>(stream);
    
    // Call kernel based on data type and index type
    // Call the appropriate template instantiation based on data types
    if (index_dtype == INFINI_DTYPE_I32) {
        switch (dtype) {
            case INFINI_DTYPE_F16:
                return calculate_scatter_cuda<fp16_t, int32_t>(
                     input_shape, output_shape, src_shape, index_shape,
                     input_strides, output_strides, index_strides, src_strides,
                     static_cast<size_t>(_dim),
                     static_cast<fp16_t*>(output),
                     static_cast<const fp16_t*>(input),
                     static_cast<const int32_t*>(index),
                     static_cast<const fp16_t*>(src),
                     cuda_stream);
                 break;
            case INFINI_DTYPE_F32:
                return calculate_scatter_cuda<float, int32_t>(
                     input_shape, output_shape, src_shape, index_shape,
                     input_strides, output_strides, index_strides, src_strides,
                     static_cast<size_t>(_dim),
                     static_cast<float*>(output),
                     static_cast<const float*>(input),
                     static_cast<const int32_t*>(index),
                     static_cast<const float*>(src),
                     cuda_stream);
                 break;
             case INFINI_DTYPE_F64:
                 return calculate_scatter_cuda<double, int32_t>(
                      input_shape, output_shape, src_shape, index_shape,
                      input_strides, output_strides, index_strides, src_strides,
                      static_cast<size_t>(_dim),
                      static_cast<double*>(output),
                      static_cast<const double*>(input),
                      static_cast<const int32_t*>(index),
                      static_cast<const double*>(src),
                      cuda_stream);
                 break;
             case INFINI_DTYPE_BF16:
                 return calculate_scatter_cuda<bf16_t, int32_t>(
                      input_shape, output_shape, src_shape, index_shape,
                      input_strides, output_strides, index_strides, src_strides,
                      static_cast<size_t>(_dim),
                      static_cast<bf16_t*>(output),
                      static_cast<const bf16_t*>(input),
                      static_cast<const int32_t*>(index),
                      static_cast<const bf16_t*>(src),
                      cuda_stream);
                 break;
             case INFINI_DTYPE_I8:
                 return calculate_scatter_cuda<int8_t, int32_t>(
                      input_shape, output_shape, src_shape, index_shape,
                      input_strides, output_strides, index_strides, src_strides,
                      static_cast<size_t>(_dim),
                      static_cast<int8_t*>(output),
                      static_cast<const int8_t*>(input),
                      static_cast<const int32_t*>(index),
                      static_cast<const int8_t*>(src),
                      cuda_stream);
                 break;
             case INFINI_DTYPE_I16:
                 return calculate_scatter_cuda<int16_t, int32_t>(
                      input_shape, output_shape, src_shape, index_shape,
                      input_strides, output_strides, index_strides, src_strides,
                      static_cast<size_t>(_dim),
                      static_cast<int16_t*>(output),
                      static_cast<const int16_t*>(input),
                      static_cast<const int32_t*>(index),
                      static_cast<const int16_t*>(src),
                      cuda_stream);
                 break;
             case INFINI_DTYPE_I32:
                 return calculate_scatter_cuda<int32_t, int32_t>(
                      input_shape, output_shape, src_shape, index_shape,
                      input_strides, output_strides, index_strides, src_strides,
                      static_cast<size_t>(_dim),
                      static_cast<int32_t*>(output),
                      static_cast<const int32_t*>(input),
                      static_cast<const int32_t*>(index),
                      static_cast<const int32_t*>(src),
                      cuda_stream);
                 break;
             case INFINI_DTYPE_I64:
                 return calculate_scatter_cuda<int64_t, int32_t>(
                      input_shape, output_shape, src_shape, index_shape,
                      input_strides, output_strides, index_strides, src_strides,
                      static_cast<size_t>(_dim),
                      static_cast<int64_t*>(output),
                      static_cast<const int64_t*>(input),
                      static_cast<const int32_t*>(index),
                      static_cast<const int64_t*>(src),
                      cuda_stream);
                 break;
             case INFINI_DTYPE_U8:
                 return calculate_scatter_cuda<uint8_t, int32_t>(
                      input_shape, output_shape, src_shape, index_shape,
                      input_strides, output_strides, index_strides, src_strides,
                      static_cast<size_t>(_dim),
                      static_cast<uint8_t*>(output),
                      static_cast<const uint8_t*>(input),
                      static_cast<const int32_t*>(index),
                      static_cast<const uint8_t*>(src),
                      cuda_stream);
                 break;
             case INFINI_DTYPE_U16:
                 return calculate_scatter_cuda<uint16_t, int32_t>(
                      input_shape, output_shape, src_shape, index_shape,
                      input_strides, output_strides, index_strides, src_strides,
                      static_cast<size_t>(_dim),
                      static_cast<uint16_t*>(output),
                      static_cast<const uint16_t*>(input),
                      static_cast<const int32_t*>(index),
                      static_cast<const uint16_t*>(src),
                      cuda_stream);
                 break;
             case INFINI_DTYPE_U32:
                 return calculate_scatter_cuda<uint32_t, int32_t>(
                      input_shape, output_shape, src_shape, index_shape,
                      input_strides, output_strides, index_strides, src_strides,
                      static_cast<size_t>(_dim),
                      static_cast<uint32_t*>(output),
                      static_cast<const uint32_t*>(input),
                      static_cast<const int32_t*>(index),
                      static_cast<const uint32_t*>(src),
                      cuda_stream);
                 break;
             case INFINI_DTYPE_U64:
                 return calculate_scatter_cuda<uint64_t, int32_t>(
                      input_shape, output_shape, src_shape, index_shape,
                      input_strides, output_strides, index_strides, src_strides,
                      static_cast<size_t>(_dim),
                      static_cast<uint64_t*>(output),
                      static_cast<const uint64_t*>(input),
                      static_cast<const int32_t*>(index),
                      static_cast<const uint64_t*>(src),
                      cuda_stream);
                 break;
             case INFINI_DTYPE_BOOL:
                 return calculate_scatter_cuda<bool, int32_t>(
                      input_shape, output_shape, src_shape, index_shape,
                      input_strides, output_strides, index_strides, src_strides,
                      static_cast<size_t>(_dim),
                      static_cast<bool*>(output),
                      static_cast<const bool*>(input),
                      static_cast<const int32_t*>(index),
                      static_cast<const bool*>(src),
                      cuda_stream);
                 break;
             default:
                 return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
    } else if (index_dtype == INFINI_DTYPE_I64) {
        switch (dtype) {
            case INFINI_DTYPE_F16:
                return calculate_scatter_cuda<fp16_t, int64_t>(
                     input_shape, output_shape, src_shape, index_shape,
                     input_strides, output_strides, index_strides, src_strides,
                     static_cast<size_t>(_dim),
                     static_cast<fp16_t*>(output),
                     static_cast<const fp16_t*>(input),
                     static_cast<const int64_t*>(index),
                     static_cast<const fp16_t*>(src),
                     cuda_stream);
                 break;
            case INFINI_DTYPE_F32:
                return calculate_scatter_cuda<float, int64_t>(
                     input_shape, output_shape, src_shape, index_shape,
                     input_strides, output_strides, index_strides, src_strides,
                     static_cast<size_t>(_dim),
                     static_cast<float*>(output),
                     static_cast<const float*>(input),
                     static_cast<const int64_t*>(index),
                     static_cast<const float*>(src),
                     cuda_stream);
                 break;
             case INFINI_DTYPE_F64:
                 return calculate_scatter_cuda<double, int64_t>(
                      input_shape, output_shape, src_shape, index_shape,
                      input_strides, output_strides, index_strides, src_strides,
                      static_cast<size_t>(_dim),
                      static_cast<double*>(output),
                      static_cast<const double*>(input),
                      static_cast<const int64_t*>(index),
                      static_cast<const double*>(src),
                      cuda_stream);
                 break;
             case INFINI_DTYPE_BF16:
                 return calculate_scatter_cuda<bf16_t, int64_t>(
                      input_shape, output_shape, src_shape, index_shape,
                      input_strides, output_strides, index_strides, src_strides,
                      static_cast<size_t>(_dim),
                      static_cast<bf16_t*>(output),
                      static_cast<const bf16_t*>(input),
                      static_cast<const int64_t*>(index),
                      static_cast<const bf16_t*>(src),
                      cuda_stream);
                 break;
             case INFINI_DTYPE_I8:
                 return calculate_scatter_cuda<int8_t, int64_t>(
                      input_shape, output_shape, src_shape, index_shape,
                      input_strides, output_strides, index_strides, src_strides,
                      static_cast<size_t>(_dim),
                      static_cast<int8_t*>(output),
                      static_cast<const int8_t*>(input),
                      static_cast<const int64_t*>(index),
                      static_cast<const int8_t*>(src),
                      cuda_stream);
                 break;
             case INFINI_DTYPE_I16:
                 return calculate_scatter_cuda<int16_t, int64_t>(
                      input_shape, output_shape, src_shape, index_shape,
                      input_strides, output_strides, index_strides, src_strides,
                      static_cast<size_t>(_dim),
                      static_cast<int16_t*>(output),
                      static_cast<const int16_t*>(input),
                      static_cast<const int64_t*>(index),
                      static_cast<const int16_t*>(src),
                      cuda_stream);
                 break;
             case INFINI_DTYPE_I32:
                 return calculate_scatter_cuda<int32_t, int64_t>(
                      input_shape, output_shape, src_shape, index_shape,
                      input_strides, output_strides, index_strides, src_strides,
                      static_cast<size_t>(_dim),
                      static_cast<int32_t*>(output),
                      static_cast<const int32_t*>(input),
                      static_cast<const int64_t*>(index),
                      static_cast<const int32_t*>(src),
                      cuda_stream);
                 break;
             case INFINI_DTYPE_I64:
                 return calculate_scatter_cuda<int64_t, int64_t>(
                      input_shape, output_shape, src_shape, index_shape,
                      input_strides, output_strides, index_strides, src_strides,
                      static_cast<size_t>(_dim),
                      static_cast<int64_t*>(output),
                      static_cast<const int64_t*>(input),
                      static_cast<const int64_t*>(index),
                      static_cast<const int64_t*>(src),
                      cuda_stream);
                 break;
             case INFINI_DTYPE_U8:
                 return calculate_scatter_cuda<uint8_t, int64_t>(
                      input_shape, output_shape, src_shape, index_shape,
                      input_strides, output_strides, index_strides, src_strides,
                      static_cast<size_t>(_dim),
                      static_cast<uint8_t*>(output),
                      static_cast<const uint8_t*>(input),
                      static_cast<const int64_t*>(index),
                      static_cast<const uint8_t*>(src),
                      cuda_stream);
                 break;
             case INFINI_DTYPE_U16:
                 return calculate_scatter_cuda<uint16_t, int64_t>(
                      input_shape, output_shape, src_shape, index_shape,
                      input_strides, output_strides, index_strides, src_strides,
                      static_cast<size_t>(_dim),
                      static_cast<uint16_t*>(output),
                      static_cast<const uint16_t*>(input),
                      static_cast<const int64_t*>(index),
                      static_cast<const uint16_t*>(src),
                      cuda_stream);
                 break;
             case INFINI_DTYPE_U32:
                 return calculate_scatter_cuda<uint32_t, int64_t>(
                      input_shape, output_shape, src_shape, index_shape,
                      input_strides, output_strides, index_strides, src_strides,
                      static_cast<size_t>(_dim),
                      static_cast<uint32_t*>(output),
                      static_cast<const uint32_t*>(input),
                      static_cast<const int64_t*>(index),
                      static_cast<const uint32_t*>(src),
                      cuda_stream);
                 break;
             case INFINI_DTYPE_U64:
                 return calculate_scatter_cuda<uint64_t, int64_t>(
                      input_shape, output_shape, src_shape, index_shape,
                      input_strides, output_strides, index_strides, src_strides,
                      static_cast<size_t>(_dim),
                      static_cast<uint64_t*>(output),
                      static_cast<const uint64_t*>(input),
                      static_cast<const int64_t*>(index),
                      static_cast<const uint64_t*>(src),
                      cuda_stream);
                 break;
             case INFINI_DTYPE_BOOL:
                 return calculate_scatter_cuda<bool, int64_t>(
                      input_shape, output_shape, src_shape, index_shape,
                      input_strides, output_strides, index_strides, src_strides,
                      static_cast<size_t>(_dim),
                      static_cast<bool*>(output),
                      static_cast<const bool*>(input),
                      static_cast<const int64_t*>(index),
                      static_cast<const bool*>(src),
                      cuda_stream);
                 break;
             default:
                 return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::scatter::nvidia
