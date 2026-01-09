#include "dot_nvidia.cuh"
#include "../cuda/kernel.cuh"
#include "../../../utils.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace op::dot::nvidia {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc) {

    auto dtype = a_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64, INFINI_DTYPE_BF16);

    // Check that y is a scalar (0D tensor or shape [1])
    auto y_shape = y_desc->shape();
    if (y_shape.size() != 0 && (y_shape.size() != 1 || y_shape[0] != 1)) {
        return INFINI_STATUS_BAD_PARAM;
    }

    // Check that a and b are 1D vectors with same length
    auto a_shape = a_desc->shape();
    auto b_shape = b_desc->shape();
    if (a_shape.size() != 1 || b_shape.size() != 1 || a_shape[0] != b_shape[0]) {
        return INFINI_STATUS_BAD_PARAM;
    }

    size_t n = a_shape[0];
    ptrdiff_t a_stride = a_desc->strides()[0];
    ptrdiff_t b_stride = b_desc->strides()[0];

    *desc_ptr = new Descriptor(dtype, n, a_stride, b_stride, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *a,
    const void *b,
    void *stream) const {

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    constexpr unsigned int BLOCK_SIZE = 256;

    // Initialize result to zero
    switch (_dtype) {
    case INFINI_DTYPE_F16: {
        half zero = __float2half(0.0f);
        CHECK_CUDA(cudaMemsetAsync(y, 0, sizeof(half), cuda_stream));
        float *result_f = nullptr;
        CHECK_CUDA(cudaMallocAsync(&result_f, sizeof(float), cuda_stream));
        CHECK_CUDA(cudaMemsetAsync(result_f, 0, sizeof(float), cuda_stream));
        cuda::dot_kernel<BLOCK_SIZE, half, float><<<1, BLOCK_SIZE, 0, cuda_stream>>>(
            result_f, reinterpret_cast<const half *>(a), reinterpret_cast<const half *>(b),
            _n, _a_stride, _b_stride);
        // Copy result back
        float result_val;
        CHECK_CUDA(cudaMemcpyAsync(&result_val, result_f, sizeof(float), cudaMemcpyDeviceToHost, cuda_stream));
        CHECK_CUDA(cudaStreamSynchronize(cuda_stream));
        *reinterpret_cast<half *>(y) = __float2half(result_val);
        CHECK_CUDA(cudaFreeAsync(result_f, cuda_stream));
        break;
    }
    case INFINI_DTYPE_BF16: {
        float *result_f = nullptr;
        CHECK_CUDA(cudaMallocAsync(&result_f, sizeof(float), cuda_stream));
        CHECK_CUDA(cudaMemsetAsync(result_f, 0, sizeof(float), cuda_stream));
        cuda::dot_kernel<BLOCK_SIZE, cuda_bfloat16, float><<<1, BLOCK_SIZE, 0, cuda_stream>>>(
            result_f, reinterpret_cast<const cuda_bfloat16 *>(a), reinterpret_cast<const cuda_bfloat16 *>(b),
            _n, _a_stride, _b_stride);
        float result_val;
        CHECK_CUDA(cudaMemcpyAsync(&result_val, result_f, sizeof(float), cudaMemcpyDeviceToHost, cuda_stream));
        CHECK_CUDA(cudaStreamSynchronize(cuda_stream));
        *reinterpret_cast<cuda_bfloat16 *>(y) = __float2bfloat16_rn(result_val);
        CHECK_CUDA(cudaFreeAsync(result_f, cuda_stream));
        break;
    }
    case INFINI_DTYPE_F32: {
        float *result_f = reinterpret_cast<float *>(y);
        CHECK_CUDA(cudaMemsetAsync(result_f, 0, sizeof(float), cuda_stream));
        cuda::dot_kernel<BLOCK_SIZE, float, float><<<1, BLOCK_SIZE, 0, cuda_stream>>>(
            result_f, reinterpret_cast<const float *>(a), reinterpret_cast<const float *>(b),
            _n, _a_stride, _b_stride);
        break;
    }
    case INFINI_DTYPE_F64: {
        double *result_d = reinterpret_cast<double *>(y);
        CHECK_CUDA(cudaMemsetAsync(result_d, 0, sizeof(double), cuda_stream));
        cuda::dot_kernel<BLOCK_SIZE, double, double><<<1, BLOCK_SIZE, 0, cuda_stream>>>(
            result_d, reinterpret_cast<const double *>(a), reinterpret_cast<const double *>(b),
            _n, _a_stride, _b_stride);
        break;
    }
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::dot::nvidia
