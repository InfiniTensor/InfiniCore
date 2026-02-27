#include "matrix_power_nvidia.cuh"
#include "../../../utils.h"
#include "../../../devices/nvidia/nvidia_handle.cuh"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <cstring>

namespace op::matrix_power::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
    
    Opaque(std::shared_ptr<device::nvidia::Handle::Internal> internal_)
        : internal(internal_) {}
};

Descriptor::~Descriptor() {
    if (_opaque) {
        delete _opaque;
    }
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    int n) {

    auto dtype = x_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64, INFINI_DTYPE_BF16);

    auto x_shape = x_desc->shape();
    auto y_shape = y_desc->shape();

    if (x_shape.size() != 2 || x_shape[0] != x_shape[1]) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    if (y_shape != x_shape) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    auto handle_nvidia = reinterpret_cast<device::nvidia::Handle *>(handle);
    Descriptor *desc = new Descriptor(dtype, x_shape[0], (n < 0) ? -n : n,
                                      x_desc->numel(), y_desc->numel(),
                                      handle->device, handle->device_id);
    desc->_opaque = new Opaque(handle_nvidia->internal());
    *desc_ptr = desc;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    void *stream) const {

    if (workspace_size < this->workspaceSize()) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    size_t n = matrix_size;
    int power = static_cast<int>(this->n);

    // Use workspace for temporary matrices
    void *temp1 = workspace;
    void *temp2 = reinterpret_cast<char *>(workspace) + n * n * infiniopGetDtypeSize(_dtype);

    size_t input_bytes = input_size * infiniopGetDtypeSize(_dtype);
    size_t output_bytes = output_size * infiniopGetDtypeSize(_dtype);
    
    // Initialize result as identity matrix
    CHECK_CUDA(cudaMemsetAsync(y, 0, output_bytes, cuda_stream));
    // Set diagonal to 1
    // TODO: Launch kernel to set identity matrix

    // Copy input to temp1
    CHECK_CUDA(cudaMemcpyAsync(temp1, x, input_bytes, cudaMemcpyDeviceToDevice, cuda_stream));

    size_t input_bytes = input_size * infiniopGetDtypeSize(_dtype);
    std::vector<float> h_matrix(input_size);
    CHECK_CUDA(cudaMemcpyAsync(h_matrix.data(), x, input_bytes, cudaMemcpyDeviceToHost, cuda_stream));
    CHECK_CUDA(cudaStreamSynchronize(cuda_stream));

    // Compute on CPU (temporary solution)
    std::vector<float> result(output_size, 0.0f);
    std::vector<float> temp1_cpu(input_size);
    std::vector<float> temp2_cpu(input_size);
    std::memcpy(temp1_cpu.data(), h_matrix.data(), input_bytes);

    // Initialize result as identity
    for (size_t i = 0; i < n; ++i) {
        result[i * n + i] = 1.0f;
    }

    // Binary exponentiation
    while (power > 0) {
        if (power & 1) {
            // Multiply result by temp1
            std::fill(temp2_cpu.begin(), temp2_cpu.end(), 0.0f);
            for (size_t i = 0; i < n; ++i) {
                for (size_t k = 0; k < n; ++k) {
                    float val = result[i * n + k];
                    for (size_t j = 0; j < n; ++j) {
                        temp2_cpu[i * n + j] += val * temp1_cpu[k * n + j];
                    }
                }
            }
            std::memcpy(result.data(), temp2_cpu.data(), output_bytes);
        }
        // Square temp1
        std::fill(temp2_cpu.begin(), temp2_cpu.end(), 0.0f);
        for (size_t i = 0; i < n; ++i) {
            for (size_t k = 0; k < n; ++k) {
                float val = temp1_cpu[i * n + k];
                for (size_t j = 0; j < n; ++j) {
                    temp2_cpu[i * n + j] += val * temp1_cpu[k * n + j];
                }
            }
        }
        std::memcpy(temp1_cpu.data(), temp2_cpu.data(), input_bytes);
        power >>= 1;
    }

    // Copy result back to GPU
    CHECK_CUDA(cudaMemcpyAsync(y, result.data(), output_bytes, cudaMemcpyHostToDevice, cuda_stream));

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::matrix_power::nvidia
