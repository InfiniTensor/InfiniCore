#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "../../../devices/nvidia/nvidia_common.cuh"
#include "linear_nvidia.cuh"
#include "../../../../utils.h"
#include "../../../tensor.h"
#include "../cuda/kernel.cuh"

namespace op::linear::nvidia {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t w_desc,
    infiniopTensorDescriptor_t b_desc,
    infiniopTensorDescriptor_t y_desc) {
    
    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    auto x_dtype = x_desc->dtype();
    auto w_dtype = w_desc->dtype();
    auto y_dtype = y_desc->dtype();
    
    // Check data types - support F16, F32, BF16
    CHECK_DTYPE(x_dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);
    
    // Check that all tensors have same dtype
    if (x_dtype != w_dtype || x_dtype != y_dtype) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    
    // Check bias data type if provided
    if (b_desc) {
        auto b_dtype = b_desc->dtype();
        if (b_dtype != x_dtype) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
    }
    
    // Check dimensions
    auto x_shape = x_desc->shape();
    auto w_shape = w_desc->shape();
    auto y_shape = y_desc->shape();
    
    int x_ndim = x_shape.size();
    int w_ndim = w_shape.size();
    int y_ndim = y_shape.size();

    if (w_ndim != 2 || x_ndim < 1 || y_ndim < 1) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    // Get dimensions
    std::vector<int> x_dims(x_shape.begin(), x_shape.end());
    std::vector<int> w_dims(w_shape.begin(), w_shape.end());
    std::vector<int> y_dims(y_shape.begin(), y_shape.end());

    // Check dimension compatibility
    // x: (..., in_features), w: (out_features, in_features), y: (..., out_features)
    int in_features = x_dims[x_ndim - 1];
    int out_features = w_dims[0];
    int w_in_features = w_dims[1];

    if (in_features != w_in_features) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    if (y_dims[y_ndim - 1] != out_features) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    // Check bias dimensions if provided
    if (b_desc) {
        auto b_shape = b_desc->shape();
        int b_ndim = b_shape.size();
        std::vector<int> b_dims(b_shape.begin(), b_shape.end());

        if (b_ndim != 1 || b_dims[0] != out_features) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
    }
    
    // Calculate batch size
    int batch_size = 1;
    for (size_t i = 0; i < x_dims.size() - 1; i++) {
        batch_size *= x_dims[i];
    }
    
    auto desc = new Descriptor();
    desc->device_type = handle->device;
    desc->device_id = handle->device_id;
    desc->_x_desc = x_desc;
    desc->_w_desc = w_desc;
    desc->_b_desc = b_desc;
    desc->_y_desc = y_desc;
    desc->_handle = handle;
    desc->_x_dims = x_dims;
    desc->_w_dims = w_dims;
    desc->_dtype = x_dtype;
    desc->_batch_size = batch_size;
    desc->_in_features = in_features;
    desc->_out_features = out_features;
    
    *desc_ptr = desc;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    const void *w,
    const void *b,
    void *stream) const {
    
    auto x_shape = _x_desc->shape();
    auto w_shape = _w_desc->shape();
    auto x_strides = _x_desc->strides();
    auto w_strides = _w_desc->strides();
    auto y_strides = _y_desc->strides();
    auto dtype = _dtype;
    
    // Calculate total output elements
    size_t total_output_elements = _batch_size * _out_features;
    
    // Copy shape and stride data to device
    int *d_x_shape, *d_w_shape;
    int *d_x_strides, *d_w_strides, *d_y_strides;
    
    int x_ndim = x_shape.size();
    int w_ndim = w_shape.size();
    
    // Declare variables before any goto statements
    int block_size = 256;
    int grid_size = (total_output_elements + block_size - 1) / block_size;
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    
    cudaError_t err;
    err = cudaMalloc(&d_x_shape, x_ndim * sizeof(int));
    if (err != cudaSuccess) return INFINI_STATUS_INTERNAL_ERROR;
    err = cudaMalloc(&d_w_shape, w_ndim * sizeof(int));
    if (err != cudaSuccess) {
        cudaFree(d_x_shape);
        return INFINI_STATUS_INTERNAL_ERROR;
    }
    err = cudaMalloc(&d_x_strides, x_ndim * sizeof(int));
    if (err != cudaSuccess) {
        cudaFree(d_x_shape);
        cudaFree(d_w_shape);
        return INFINI_STATUS_INTERNAL_ERROR;
    }
    err = cudaMalloc(&d_w_strides, w_ndim * sizeof(int));
    if (err != cudaSuccess) {
        cudaFree(d_x_shape);
        cudaFree(d_w_shape);
        cudaFree(d_x_strides);
        return INFINI_STATUS_INTERNAL_ERROR;
    }
    err = cudaMalloc(&d_y_strides, x_ndim * sizeof(int)); // y has same ndim as x
    if (err != cudaSuccess) {
        cudaFree(d_x_shape);
        cudaFree(d_w_shape);
        cudaFree(d_x_strides);
        cudaFree(d_w_strides);
        return INFINI_STATUS_INTERNAL_ERROR;
    }
    
    std::vector<int> h_x_shape(x_shape.begin(), x_shape.end());
    std::vector<int> h_w_shape(w_shape.begin(), w_shape.end());
    std::vector<int> h_x_strides(x_strides.begin(), x_strides.end());
    std::vector<int> h_w_strides(w_strides.begin(), w_strides.end());
    std::vector<int> h_y_strides(y_strides.begin(), y_strides.end());
    
    err = cudaMemcpy(d_x_shape, h_x_shape.data(), x_ndim * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup;
    err = cudaMemcpy(d_w_shape, h_w_shape.data(), w_ndim * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup;
    err = cudaMemcpy(d_x_strides, h_x_strides.data(), x_ndim * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup;
    err = cudaMemcpy(d_w_strides, h_w_strides.data(), w_ndim * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup;
    err = cudaMemcpy(d_y_strides, h_y_strides.data(), x_ndim * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup;
    
    // Dispatch based on data type
    switch (dtype) {
        case INFINI_DTYPE_F16:
            cuda::linear_kernel<__half><<<grid_size, block_size, 0, cuda_stream>>>(
                static_cast<__half*>(y),
                static_cast<const __half*>(x),
                static_cast<const __half*>(w),
                static_cast<const __half*>(b),
                d_x_shape, d_w_shape,
                d_x_strides, d_w_strides, d_y_strides,
                _batch_size, _in_features, _out_features,
                total_output_elements);
            break;
        case INFINI_DTYPE_F32:
            cuda::linear_kernel<float><<<grid_size, block_size, 0, cuda_stream>>>(
                static_cast<float*>(y),
                static_cast<const float*>(x),
                static_cast<const float*>(w),
                static_cast<const float*>(b),
                d_x_shape, d_w_shape,
                d_x_strides, d_w_strides, d_y_strides,
                _batch_size, _in_features, _out_features,
                total_output_elements);
            break;
        case INFINI_DTYPE_BF16:
            cuda::linear_kernel<__nv_bfloat16><<<grid_size, block_size, 0, cuda_stream>>>(
                static_cast<__nv_bfloat16*>(y),
                static_cast<const __nv_bfloat16*>(x),
                static_cast<const __nv_bfloat16*>(w),
                static_cast<const __nv_bfloat16*>(b),
                d_x_shape, d_w_shape,
                d_x_strides, d_w_strides, d_y_strides,
                _batch_size, _in_features, _out_features,
                total_output_elements);
            break;
        default:
            err = cudaErrorInvalidValue;
            goto cleanup;
    }
    
    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) goto cleanup;
    
    // Wait for kernel to complete
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) goto cleanup;
    
cleanup:
    // Cleanup device memory
    cudaFree(d_x_shape);
    cudaFree(d_w_shape);
    cudaFree(d_x_strides);
    cudaFree(d_w_strides);
    cudaFree(d_y_strides);
    
    return (err == cudaSuccess) ? INFINI_STATUS_SUCCESS : INFINI_STATUS_INTERNAL_ERROR;
}

} // namespace op::linear::nvidia