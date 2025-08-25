#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "../../../devices/nvidia/nvidia_common.cuh"
#include "index_copy_inplace_nvidia.cuh"
#include "../../../../utils.h"
#include "../../../tensor.h"
#include "../cuda/kernel.cuh"

namespace op::index_copy_inplace::nvidia {

Descriptor::~Descriptor() = default;



infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t output_desc,
    int dim,
    infiniopTensorDescriptor_t index_desc) {
    
    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    auto dtype = input_desc->dtype();
    
    // Check data types - 支持所有合法类型
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64, INFINI_DTYPE_BF16,
                INFINI_DTYPE_I8, INFINI_DTYPE_I16, INFINI_DTYPE_I32, INFINI_DTYPE_I64,
                INFINI_DTYPE_U8, INFINI_DTYPE_U16, INFINI_DTYPE_U32, INFINI_DTYPE_U64,
                INFINI_DTYPE_BOOL);
    
    // Check that input and output have same dtype
    if (input_desc->dtype() != output_desc->dtype()) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    
    // Check that index is integer type
    auto index_dtype = index_desc->dtype();
    if (index_dtype != INFINI_DTYPE_I32 && index_dtype != INFINI_DTYPE_I64) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    
    // Check dimension bounds
    auto input_shape = input_desc->shape();
    auto output_shape = output_desc->shape();
    if (dim < 0 || dim >= static_cast<int>(input_shape.size())) {
        return INFINI_STATUS_BAD_PARAM;
    }
    if (dim < 0 || dim >= static_cast<int>(output_shape.size())) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    // Check that input and output have same shape except possibly at dim
    if (input_shape.size() != output_shape.size()) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }
    
    auto desc = new Descriptor();
    desc->device_type = handle->device;  // 设置设备类型
    desc->device_id = handle->device_id;  // 设置设备ID
    desc->_input_desc = input_desc;
    desc->_output_desc = output_desc;
    desc->_index_desc = index_desc;
    desc->_dim = dim;
    desc->_handle = handle;
    
    *desc_ptr = desc;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *input,
    void *output,
    const void *index,
    void *stream) const {
    
    auto input_shape = _input_desc->shape();
    auto output_shape = _output_desc->shape();
    auto index_shape = _index_desc->shape();
    auto input_strides = _input_desc->strides();
    auto output_strides = _output_desc->strides();
    auto index_strides = _index_desc->strides();
    auto dtype = _input_desc->dtype();
    auto index_dtype = _index_desc->dtype();
    
    // Calculate total elements based on actual input shape
    size_t total_elements = 1;
    for (size_t s : input_shape) {
        total_elements *= s;
    }
    
    // Copy shape and stride data to device
    int *d_input_shape, *d_output_shape, *d_index_shape;
    int *d_input_strides, *d_output_strides, *d_index_strides;
    
    int ndim = input_shape.size();
     
     // Declare variables before any goto statements
     int block_size = 256;
     int grid_size = (total_elements + block_size - 1) / block_size;
     cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
     
     cudaError_t err;
     err = cudaMalloc(&d_input_shape, ndim * sizeof(int));
    if (err != cudaSuccess) return INFINI_STATUS_INTERNAL_ERROR;
     err = cudaMalloc(&d_output_shape, ndim * sizeof(int));
     if (err != cudaSuccess) {
         cudaFree(d_input_shape);
         return INFINI_STATUS_INTERNAL_ERROR;
     }
     err = cudaMalloc(&d_index_shape, sizeof(int));
     if (err != cudaSuccess) {
         cudaFree(d_input_shape);
         cudaFree(d_output_shape);
         return INFINI_STATUS_INTERNAL_ERROR;
     }
     err = cudaMalloc(&d_input_strides, ndim * sizeof(int));
     if (err != cudaSuccess) {
         cudaFree(d_input_shape);
         cudaFree(d_output_shape);
         cudaFree(d_index_shape);
         return INFINI_STATUS_INTERNAL_ERROR;
     }
     err = cudaMalloc(&d_output_strides, ndim * sizeof(int));
     if (err != cudaSuccess) {
         cudaFree(d_input_shape);
         cudaFree(d_output_shape);
         cudaFree(d_index_shape);
         cudaFree(d_input_strides);
         return INFINI_STATUS_INTERNAL_ERROR;
     }
     err = cudaMalloc(&d_index_strides, sizeof(int));
     if (err != cudaSuccess) {
         cudaFree(d_input_shape);
         cudaFree(d_output_shape);
         cudaFree(d_index_shape);
         cudaFree(d_input_strides);
         cudaFree(d_output_strides);
         return INFINI_STATUS_INTERNAL_ERROR;
     }
    
    std::vector<int> h_input_shape(input_shape.begin(), input_shape.end());
    std::vector<int> h_output_shape(output_shape.begin(), output_shape.end());
    std::vector<int> h_input_strides(input_strides.begin(), input_strides.end());
    std::vector<int> h_output_strides(output_strides.begin(), output_strides.end());
    int h_index_shape = index_shape[0];
    int h_index_stride = index_strides[0];
    
    err = cudaMemcpy(d_input_shape, h_input_shape.data(), ndim * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup;
    err = cudaMemcpy(d_output_shape, h_output_shape.data(), ndim * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup;
    err = cudaMemcpy(d_index_shape, &h_index_shape, sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup;
    err = cudaMemcpy(d_input_strides, h_input_strides.data(), ndim * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup;
    err = cudaMemcpy(d_output_strides, h_output_strides.data(), ndim * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup;
    err = cudaMemcpy(d_index_strides, &h_index_stride, sizeof(int), cudaMemcpyHostToDevice);
     if (err != cudaSuccess) goto cleanup;
    
    // Dispatch based on data type and index type
    if (index_dtype == INFINI_DTYPE_I32) {
        switch (dtype) {
            case INFINI_DTYPE_F16:
                cuda::index_copy_inplace_kernel<__half, int32_t><<<grid_size, block_size, 0, cuda_stream>>>(
                    static_cast<const __half*>(input),
                    static_cast<__half*>(output),
                    static_cast<const int32_t*>(index),
                    d_input_shape, d_output_shape, d_index_shape,
                    d_input_strides, d_output_strides, d_index_strides,
                    _dim, ndim, total_elements);
                break;
            case INFINI_DTYPE_F32:
                cuda::index_copy_inplace_kernel<float, int32_t><<<grid_size, block_size, 0, cuda_stream>>>(
                    static_cast<const float*>(input),
                    static_cast<float*>(output),
                    static_cast<const int32_t*>(index),
                    d_input_shape, d_output_shape, d_index_shape,
                    d_input_strides, d_output_strides, d_index_strides,
                    _dim, ndim, total_elements);
                break;
            case INFINI_DTYPE_F64:
                cuda::index_copy_inplace_kernel<double, int32_t><<<grid_size, block_size, 0, cuda_stream>>>(
                    static_cast<const double*>(input),
                    static_cast<double*>(output),
                    static_cast<const int32_t*>(index),
                    d_input_shape, d_output_shape, d_index_shape,
                    d_input_strides, d_output_strides, d_index_strides,
                    _dim, ndim, total_elements);
                break;
            case INFINI_DTYPE_BF16:
                cuda::index_copy_inplace_kernel<__nv_bfloat16, int32_t><<<grid_size, block_size, 0, cuda_stream>>>(
                    static_cast<const __nv_bfloat16*>(input),
                    static_cast<__nv_bfloat16*>(output),
                    static_cast<const int32_t*>(index),
                    d_input_shape, d_output_shape, d_index_shape,
                    d_input_strides, d_output_strides, d_index_strides,
                    _dim, ndim, total_elements);
                break;
            case INFINI_DTYPE_I8:
                cuda::index_copy_inplace_kernel<int8_t, int32_t><<<grid_size, block_size, 0, cuda_stream>>>(
                    static_cast<const int8_t*>(input),
                    static_cast<int8_t*>(output),
                    static_cast<const int32_t*>(index),
                    d_input_shape, d_output_shape, d_index_shape,
                    d_input_strides, d_output_strides, d_index_strides,
                    _dim, ndim, total_elements);
                break;
            case INFINI_DTYPE_I16:
                cuda::index_copy_inplace_kernel<int16_t, int32_t><<<grid_size, block_size, 0, cuda_stream>>>(
                    static_cast<const int16_t*>(input),
                    static_cast<int16_t*>(output),
                    static_cast<const int32_t*>(index),
                    d_input_shape, d_output_shape, d_index_shape,
                    d_input_strides, d_output_strides, d_index_strides,
                    _dim, ndim, total_elements);
                break;
            case INFINI_DTYPE_I32:
                cuda::index_copy_inplace_kernel<int32_t, int32_t><<<grid_size, block_size, 0, cuda_stream>>>(
                    static_cast<const int32_t*>(input),
                    static_cast<int32_t*>(output),
                    static_cast<const int32_t*>(index),
                    d_input_shape, d_output_shape, d_index_shape,
                    d_input_strides, d_output_strides, d_index_strides,
                    _dim, ndim, total_elements);
                break;
            case INFINI_DTYPE_I64:
                cuda::index_copy_inplace_kernel<int64_t, int32_t><<<grid_size, block_size, 0, cuda_stream>>>(
                    static_cast<const int64_t*>(input),
                    static_cast<int64_t*>(output),
                    static_cast<const int32_t*>(index),
                    d_input_shape, d_output_shape, d_index_shape,
                    d_input_strides, d_output_strides, d_index_strides,
                    _dim, ndim, total_elements);
                break;
            case INFINI_DTYPE_U8:
                cuda::index_copy_inplace_kernel<uint8_t, int32_t><<<grid_size, block_size, 0, cuda_stream>>>(
                    static_cast<const uint8_t*>(input),
                    static_cast<uint8_t*>(output),
                    static_cast<const int32_t*>(index),
                    d_input_shape, d_output_shape, d_index_shape,
                    d_input_strides, d_output_strides, d_index_strides,
                    _dim, ndim, total_elements);
                break;
            case INFINI_DTYPE_U16:
                cuda::index_copy_inplace_kernel<uint16_t, int32_t><<<grid_size, block_size, 0, cuda_stream>>>(
                    static_cast<const uint16_t*>(input),
                    static_cast<uint16_t*>(output),
                    static_cast<const int32_t*>(index),
                    d_input_shape, d_output_shape, d_index_shape,
                    d_input_strides, d_output_strides, d_index_strides,
                    _dim, ndim, total_elements);
                break;
            case INFINI_DTYPE_U32:
                cuda::index_copy_inplace_kernel<uint32_t, int32_t><<<grid_size, block_size, 0, cuda_stream>>>(
                    static_cast<const uint32_t*>(input),
                    static_cast<uint32_t*>(output),
                    static_cast<const int32_t*>(index),
                    d_input_shape, d_output_shape, d_index_shape,
                    d_input_strides, d_output_strides, d_index_strides,
                    _dim, ndim, total_elements);
                break;
            case INFINI_DTYPE_U64:
                cuda::index_copy_inplace_kernel<uint64_t, int32_t><<<grid_size, block_size, 0, cuda_stream>>>(
                    static_cast<const uint64_t*>(input),
                    static_cast<uint64_t*>(output),
                    static_cast<const int32_t*>(index),
                    d_input_shape, d_output_shape, d_index_shape,
                    d_input_strides, d_output_strides, d_index_strides,
                    _dim, ndim, total_elements);
                break;
            case INFINI_DTYPE_BOOL:
                cuda::index_copy_inplace_kernel<bool, int32_t><<<grid_size, block_size, 0, cuda_stream>>>(
                    static_cast<const bool*>(input),
                    static_cast<bool*>(output),
                    static_cast<const int32_t*>(index),
                    d_input_shape, d_output_shape, d_index_shape,
                    d_input_strides, d_output_strides, d_index_strides,
                    _dim, ndim, total_elements);
                break;
            default:
                return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
    } else if (index_dtype == INFINI_DTYPE_I64) {
        // Similar dispatch for int64_t index type
        switch (dtype) {
            case INFINI_DTYPE_F16:
                cuda::index_copy_inplace_kernel<__half, int64_t><<<grid_size, block_size, 0, cuda_stream>>>(
                    static_cast<const __half*>(input),
                    static_cast<__half*>(output),
                    static_cast<const int64_t*>(index),
                    d_input_shape, d_output_shape, d_index_shape,
                    d_input_strides, d_output_strides, d_index_strides,
                    _dim, ndim, total_elements);
                break;
            case INFINI_DTYPE_F32:
                cuda::index_copy_inplace_kernel<float, int64_t><<<grid_size, block_size, 0, cuda_stream>>>(
                    static_cast<const float*>(input),
                    static_cast<float*>(output),
                    static_cast<const int64_t*>(index),
                    d_input_shape, d_output_shape, d_index_shape,
                    d_input_strides, d_output_strides, d_index_strides,
                    _dim, ndim, total_elements);
                break;
            case INFINI_DTYPE_F64:
                cuda::index_copy_inplace_kernel<double, int64_t><<<grid_size, block_size, 0, cuda_stream>>>(
                    static_cast<const double*>(input),
                    static_cast<double*>(output),
                    static_cast<const int64_t*>(index),
                    d_input_shape, d_output_shape, d_index_shape,
                    d_input_strides, d_output_strides, d_index_strides,
                    _dim, ndim, total_elements);
                break;
            case INFINI_DTYPE_BF16:
                cuda::index_copy_inplace_kernel<__nv_bfloat16, int64_t><<<grid_size, block_size, 0, cuda_stream>>>(
                    static_cast<const __nv_bfloat16*>(input),
                    static_cast<__nv_bfloat16*>(output),
                    static_cast<const int64_t*>(index),
                    d_input_shape, d_output_shape, d_index_shape,
                    d_input_strides, d_output_strides, d_index_strides,
                    _dim, ndim, total_elements);
                break;
            case INFINI_DTYPE_I8:
                cuda::index_copy_inplace_kernel<int8_t, int64_t><<<grid_size, block_size, 0, cuda_stream>>>(
                    static_cast<const int8_t*>(input),
                    static_cast<int8_t*>(output),
                    static_cast<const int64_t*>(index),
                    d_input_shape, d_output_shape, d_index_shape,
                    d_input_strides, d_output_strides, d_index_strides,
                    _dim, ndim, total_elements);
                break;
            case INFINI_DTYPE_I16:
                cuda::index_copy_inplace_kernel<int16_t, int64_t><<<grid_size, block_size, 0, cuda_stream>>>(
                    static_cast<const int16_t*>(input),
                    static_cast<int16_t*>(output),
                    static_cast<const int64_t*>(index),
                    d_input_shape, d_output_shape, d_index_shape,
                    d_input_strides, d_output_strides, d_index_strides,
                    _dim, ndim, total_elements);
                break;
            case INFINI_DTYPE_I32:
                cuda::index_copy_inplace_kernel<int32_t, int64_t><<<grid_size, block_size, 0, cuda_stream>>>(
                    static_cast<const int32_t*>(input),
                    static_cast<int32_t*>(output),
                    static_cast<const int64_t*>(index),
                    d_input_shape, d_output_shape, d_index_shape,
                    d_input_strides, d_output_strides, d_index_strides,
                    _dim, ndim, total_elements);
                break;
            case INFINI_DTYPE_I64:
                cuda::index_copy_inplace_kernel<int64_t, int64_t><<<grid_size, block_size, 0, cuda_stream>>>(
                    static_cast<const int64_t*>(input),
                    static_cast<int64_t*>(output),
                    static_cast<const int64_t*>(index),
                    d_input_shape, d_output_shape, d_index_shape,
                    d_input_strides, d_output_strides, d_index_strides,
                    _dim, ndim, total_elements);
                break;
            case INFINI_DTYPE_U8:
                cuda::index_copy_inplace_kernel<uint8_t, int64_t><<<grid_size, block_size, 0, cuda_stream>>>(
                    static_cast<const uint8_t*>(input),
                    static_cast<uint8_t*>(output),
                    static_cast<const int64_t*>(index),
                    d_input_shape, d_output_shape, d_index_shape,
                    d_input_strides, d_output_strides, d_index_strides,
                    _dim, ndim, total_elements);
                break;
            case INFINI_DTYPE_U16:
                cuda::index_copy_inplace_kernel<uint16_t, int64_t><<<grid_size, block_size, 0, cuda_stream>>>(
                    static_cast<const uint16_t*>(input),
                    static_cast<uint16_t*>(output),
                    static_cast<const int64_t*>(index),
                    d_input_shape, d_output_shape, d_index_shape,
                    d_input_strides, d_output_strides, d_index_strides,
                    _dim, ndim, total_elements);
                break;
            case INFINI_DTYPE_U32:
                cuda::index_copy_inplace_kernel<uint32_t, int64_t><<<grid_size, block_size, 0, cuda_stream>>>(
                    static_cast<const uint32_t*>(input),
                    static_cast<uint32_t*>(output),
                    static_cast<const int64_t*>(index),
                    d_input_shape, d_output_shape, d_index_shape,
                    d_input_strides, d_output_strides, d_index_strides,
                    _dim, ndim, total_elements);
                break;
            case INFINI_DTYPE_U64:
                cuda::index_copy_inplace_kernel<uint64_t, int64_t><<<grid_size, block_size, 0, cuda_stream>>>(
                    static_cast<const uint64_t*>(input),
                    static_cast<uint64_t*>(output),
                    static_cast<const int64_t*>(index),
                    d_input_shape, d_output_shape, d_index_shape,
                    d_input_strides, d_output_strides, d_index_strides,
                    _dim, ndim, total_elements);
                break;
            case INFINI_DTYPE_BOOL:
                cuda::index_copy_inplace_kernel<bool, int64_t><<<grid_size, block_size, 0, cuda_stream>>>(
                    static_cast<const bool*>(input),
                    static_cast<bool*>(output),
                    static_cast<const int64_t*>(index),
                    d_input_shape, d_output_shape, d_index_shape,
                    d_input_strides, d_output_strides, d_index_strides,
                    _dim, ndim, total_elements);
                break;
            default:
                return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
    }
    
    // Check for kernel launch errors
    err = cudaGetLastError();
    printf("DEBUG: Kernel launch result: %s\n", cudaGetErrorString(err));
    if (err != cudaSuccess) goto cleanup;
    
    // Wait for kernel to complete
    err = cudaDeviceSynchronize();
    printf("DEBUG: Kernel synchronization result: %s\n", cudaGetErrorString(err));
    if (err != cudaSuccess) goto cleanup;
    
cleanup:
    // Cleanup device memory
    cudaFree(d_input_shape);
    cudaFree(d_output_shape);
    cudaFree(d_index_shape);
    cudaFree(d_input_strides);
    cudaFree(d_output_strides);
    cudaFree(d_index_strides);
    
    return (err == cudaSuccess) ? INFINI_STATUS_SUCCESS : INFINI_STATUS_INTERNAL_ERROR;
}

} // namespace op::index_copy_inplace::nvidia