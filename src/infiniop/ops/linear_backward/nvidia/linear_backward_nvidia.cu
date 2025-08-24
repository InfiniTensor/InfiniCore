#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "../../../devices/nvidia/nvidia_common.cuh"
#include "linear_backward_nvidia.cuh"
#include "../../../../utils.h"
#include "../../../tensor.h"
#include "../cuda/kernel.cuh"

namespace op::linear_backward::nvidia {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t grad_y_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t w_desc,
    infiniopTensorDescriptor_t grad_x_desc,
    infiniopTensorDescriptor_t grad_w_desc,
    infiniopTensorDescriptor_t grad_b_desc) {
    
    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    auto grad_y_dtype = grad_y_desc->dtype();
    auto x_dtype = x_desc->dtype();
    auto w_dtype = w_desc->dtype();
    
    // Check data types - support F16, F32, BF16
    CHECK_DTYPE(grad_y_dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);
    
    // Check that all input tensors have same dtype
    if (grad_y_dtype != x_dtype || grad_y_dtype != w_dtype) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    
    // Check gradient tensor data types if provided
    if (grad_x_desc && grad_x_desc->dtype() != grad_y_dtype) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    if (grad_w_desc && grad_w_desc->dtype() != grad_y_dtype) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    if (grad_b_desc && grad_b_desc->dtype() != grad_y_dtype) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    
    // Check dimensions
    auto grad_y_shape = grad_y_desc->shape();
    auto x_shape = x_desc->shape();
    auto w_shape = w_desc->shape();
    
    int grad_y_ndim = grad_y_shape.size();
    int x_ndim = x_shape.size();
    int w_ndim = w_shape.size();

    if (w_ndim != 2 || x_ndim < 1 || grad_y_ndim < 1) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    // Get dimensions
    std::vector<int> grad_y_dims(grad_y_shape.begin(), grad_y_shape.end());
    std::vector<int> x_dims(x_shape.begin(), x_shape.end());
    std::vector<int> w_dims(w_shape.begin(), w_shape.end());

    // Check dimension compatibility
    // x: (..., in_features), w: (out_features, in_features), grad_y: (..., out_features)
    int in_features = x_dims[x_ndim - 1];
    int out_features = w_dims[0];
    int w_in_features = w_dims[1];

    if (in_features != w_in_features) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    if (grad_y_dims[grad_y_ndim - 1] != out_features) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    // Check gradient tensor dimensions if provided
    if (grad_x_desc) {
        auto grad_x_shape = grad_x_desc->shape();
        if (grad_x_shape != x_shape) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
    }
    
    if (grad_w_desc) {
        auto grad_w_shape = grad_w_desc->shape();
        if (grad_w_shape != w_shape) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
    }
    
    if (grad_b_desc) {
        auto grad_b_shape = grad_b_desc->shape();
        int grad_b_ndim = grad_b_shape.size();
        std::vector<int> grad_b_dims(grad_b_shape.begin(), grad_b_shape.end());

        if (grad_b_ndim != 1 || grad_b_dims[0] != out_features) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
    }

    // Calculate batch size
    int batch_size = 1;
    for (int i = 0; i < grad_y_ndim - 1; i++) {
        batch_size *= grad_y_dims[i];
    }

    // Create descriptor
    auto desc = new Descriptor();
    desc->device_type = handle->device;
    desc->device_id = handle->device_id;
    desc->_grad_y_desc = grad_y_desc;
    desc->_x_desc = x_desc;
    desc->_w_desc = w_desc;
    desc->_grad_x_desc = grad_x_desc;
    desc->_grad_w_desc = grad_w_desc;
    desc->_grad_b_desc = grad_b_desc;
    desc->_handle = handle;
    desc->_grad_y_dims = grad_y_dims;
    desc->_x_dims = x_dims;
    desc->_w_dims = w_dims;
    desc->_dtype = grad_y_dtype;
    desc->_batch_size = batch_size;
    desc->_in_features = in_features;
    desc->_out_features = out_features;

    *desc_ptr = desc;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *grad_x,
    void *grad_w,
    void *grad_b,
    const void *grad_y,
    const void *x,
    const void *w,
    void *stream) const {

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    
    // Check for null descriptors
    if (!_grad_y_desc || !_x_desc || !_w_desc) {
        return INFINI_STATUS_INTERNAL_ERROR;
    }
    
    // Get tensor shapes and strides
    auto grad_y_shape = _grad_y_desc->shape();
    auto x_shape = _x_desc->shape();
    auto w_shape = _w_desc->shape();
    
    auto grad_y_strides = _grad_y_desc->strides();
    auto x_strides = _x_desc->strides();
    auto w_strides = _w_desc->strides();
    
    // Declare all variables at the beginning to avoid goto issues
    int *d_grad_y_shape = nullptr, *d_x_shape = nullptr, *d_w_shape = nullptr;
    int *d_grad_y_strides = nullptr, *d_x_strides = nullptr, *d_w_strides = nullptr;
    int *d_grad_x_strides = nullptr, *d_grad_w_strides = nullptr, *d_grad_b_strides = nullptr;
    
    // Convert shape and stride data to vectors for copying
    std::vector<int> h_grad_y_shape(grad_y_shape.begin(), grad_y_shape.end());
    std::vector<int> h_x_shape(x_shape.begin(), x_shape.end());
    std::vector<int> h_w_shape(w_shape.begin(), w_shape.end());
    std::vector<int> h_grad_y_strides(grad_y_strides.begin(), grad_y_strides.end());
    std::vector<int> h_x_strides(x_strides.begin(), x_strides.end());
    std::vector<int> h_w_strides(w_strides.begin(), w_strides.end());
    
    // Calculate grid and block dimensions
    int max_elements = std::max(std::max(_batch_size * _in_features, _out_features * _in_features), _out_features);
    dim3 block(256);
    dim3 grid((max_elements + block.x - 1) / block.x);
    
    cudaError_t err = cudaSuccess;
    
    // Allocate device memory for shapes and strides
    err = cudaMalloc(&d_grad_y_shape, grad_y_shape.size() * sizeof(int));
    if (err != cudaSuccess) return INFINI_STATUS_INTERNAL_ERROR;
    err = cudaMalloc(&d_x_shape, x_shape.size() * sizeof(int));
    if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc(&d_w_shape, w_shape.size() * sizeof(int));
    if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc(&d_grad_y_strides, grad_y_strides.size() * sizeof(int));
    if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc(&d_x_strides, x_strides.size() * sizeof(int));
    if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc(&d_w_strides, w_strides.size() * sizeof(int));
    if (err != cudaSuccess) goto cleanup;
    
    // Copy data to device
    err = cudaMemcpy(d_grad_y_shape, h_grad_y_shape.data(), grad_y_shape.size() * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup;
    err = cudaMemcpy(d_x_shape, h_x_shape.data(), x_shape.size() * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup;
    err = cudaMemcpy(d_w_shape, h_w_shape.data(), w_shape.size() * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup;
    err = cudaMemcpy(d_grad_y_strides, h_grad_y_strides.data(), grad_y_strides.size() * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup;
    err = cudaMemcpy(d_x_strides, h_x_strides.data(), x_strides.size() * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup;
    err = cudaMemcpy(d_w_strides, h_w_strides.data(), w_strides.size() * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup;
    
    // Allocate strides for gradient tensors if they exist
    if (grad_x && _grad_x_desc) {
        auto grad_x_strides = _grad_x_desc->strides();
        err = cudaMalloc(&d_grad_x_strides, grad_x_strides.size() * sizeof(int));
        if (err != cudaSuccess) goto cleanup;
        std::vector<int> h_grad_x_strides(grad_x_strides.begin(), grad_x_strides.end());
        err = cudaMemcpy(d_grad_x_strides, h_grad_x_strides.data(), grad_x_strides.size() * sizeof(int), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) goto cleanup;
    }
    
    if (grad_w && _grad_w_desc) {
        auto grad_w_strides = _grad_w_desc->strides();
        err = cudaMalloc(&d_grad_w_strides, grad_w_strides.size() * sizeof(int));
        if (err != cudaSuccess) goto cleanup;
        std::vector<int> h_grad_w_strides(grad_w_strides.begin(), grad_w_strides.end());
        err = cudaMemcpy(d_grad_w_strides, h_grad_w_strides.data(), grad_w_strides.size() * sizeof(int), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) goto cleanup;
    }
    
    if (grad_b && _grad_b_desc) {
        auto grad_b_strides = _grad_b_desc->strides();
        err = cudaMalloc(&d_grad_b_strides, grad_b_strides.size() * sizeof(int));
        if (err != cudaSuccess) goto cleanup;
        std::vector<int> h_grad_b_strides(grad_b_strides.begin(), grad_b_strides.end());
        err = cudaMemcpy(d_grad_b_strides, h_grad_b_strides.data(), grad_b_strides.size() * sizeof(int), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) goto cleanup;
    }
    
    // Launch kernel based on data type
    
    if (_dtype == INFINI_DTYPE_F32) {
        op::linear_backward::cuda::linear_backward_kernel<<<grid, block, 0, cuda_stream>>>(
            grad_x,
            grad_w,
            grad_b,
            grad_y,
            x,
            w,
            _batch_size, _in_features, _out_features,
            INFINI_DTYPE_F32);
    } else if (_dtype == INFINI_DTYPE_F16) {
        op::linear_backward::cuda::linear_backward_kernel<<<grid, block, 0, cuda_stream>>>(
            grad_x,
            grad_w,
            grad_b,
            grad_y,
            x,
            w,
            _batch_size, _in_features, _out_features,
            INFINI_DTYPE_F16);
    } else if (_dtype == INFINI_DTYPE_BF16) {
        op::linear_backward::cuda::linear_backward_kernel<<<grid, block, 0, cuda_stream>>>(
            grad_x,
            grad_w,
            grad_b,
            grad_y,
            x,
            w,
            _batch_size, _in_features, _out_features,
            INFINI_DTYPE_BF16);
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    
    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) goto cleanup;
    
    // Wait for kernel to complete
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) goto cleanup;

cleanup:
    // Cleanup device memory
    if (d_grad_y_shape) cudaFree(d_grad_y_shape);
    if (d_x_shape) cudaFree(d_x_shape);
    if (d_w_shape) cudaFree(d_w_shape);
    if (d_grad_y_strides) cudaFree(d_grad_y_strides);
    if (d_x_strides) cudaFree(d_x_strides);
    if (d_w_strides) cudaFree(d_w_strides);
    if (d_grad_x_strides) cudaFree(d_grad_x_strides);
    if (d_grad_w_strides) cudaFree(d_grad_w_strides);
    if (d_grad_b_strides) cudaFree(d_grad_b_strides);
    
    return (err == cudaSuccess) ? INFINI_STATUS_SUCCESS : INFINI_STATUS_INTERNAL_ERROR;
}

} // namespace op::linear_backward::nvidia