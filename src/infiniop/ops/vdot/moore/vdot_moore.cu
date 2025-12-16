#include "../../../devices/moore/moore_handle.h"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "vdot_moore.h"
#include <cub/block/block_reduce.cuh>
#include <cuda_runtime.h>

namespace op::vdot::moore {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(infiniopHandle_t handle_,
                                  Descriptor **desc_ptr,
                                  infiniopTensorDescriptor_t out_desc,
                                  infiniopTensorDescriptor_t a_desc,
                                  infiniopTensorDescriptor_t b_desc) {

    auto handle = reinterpret_cast<device::moore::Handle *>(handle_);
    auto in_dtype = a_desc->dtype();
    auto b_dtype = b_desc->dtype();
    auto out_dtype = out_desc->dtype();

    // Inputs must be 1D vectors with same length
    if (a_desc->ndim() != 1 || b_desc->ndim() != 1) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }
    if (a_desc->numel() != b_desc->numel()) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    // Input dtypes must match and be in supported set
    CHECK_OR_RETURN(in_dtype == b_dtype, INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_DTYPE(in_dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64,
                INFINI_DTYPE_BF16);

    // Output dtype equals input dtype
    CHECK_OR_RETURN(out_dtype == in_dtype, INFINI_STATUS_BAD_TENSOR_DTYPE);

    size_t length = a_desc->numel();
    ptrdiff_t a_stride = a_desc->stride(0);
    ptrdiff_t b_stride = b_desc->stride(0);

    *desc_ptr = new Descriptor(in_dtype, out_dtype, length, a_stride, b_stride,
                               handle->device, handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(void *workspace, size_t workspace_size,
                                     void *out, const void *a, const void *b,
                                     void *stream) const {

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    constexpr unsigned int BLOCK_SIZE = 256;

    switch (_in_dtype) {
    case INFINI_DTYPE_F32: {
        float *out_f = reinterpret_cast<float *>(out);
        const float *a_f = reinterpret_cast<const float *>(a);
        const float *b_f = reinterpret_cast<const float *>(b);
        op::vdot::cuda::vdotKernel<BLOCK_SIZE, float, float>
            <<<1, BLOCK_SIZE, 0, cuda_stream>>>(out_f, a_f, b_f, _length, _a_stride,
                                                _b_stride);
        CHECK_CUDA(cudaGetLastError());
        break;
    }
    case INFINI_DTYPE_F64: {
        double *out_d = reinterpret_cast<double *>(out);
        const double *a_d = reinterpret_cast<const double *>(a);
        const double *b_d = reinterpret_cast<const double *>(b);
        op::vdot::cuda::vdotKernel<BLOCK_SIZE, double, double>
            <<<1, BLOCK_SIZE, 0, cuda_stream>>>(out_d, a_d, b_d, _length, _a_stride,
                                                _b_stride);
        CHECK_CUDA(cudaGetLastError());
        break;
    }
    case INFINI_DTYPE_F16: {
        // For FP16, accumulate in float, then cast back to half
        // Use workspace for temporary float buffer
        if (workspace_size < sizeof(float)) {
            return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
        }
        float *tmp_out = reinterpret_cast<float *>(workspace);
        {
            // If workspace is too small, we need to allocate
            // For simplicity, use a device-side kernel that writes directly to out
            // But we need float accumulation, so use a temporary approach
            const __half *a_h = reinterpret_cast<const __half *>(a);
            const __half *b_h = reinterpret_cast<const __half *>(b);
            // Launch kernel that accumulates in float and writes half result
            op::vdot::cuda::vdotKernel<BLOCK_SIZE, __half, float>
                <<<1, BLOCK_SIZE, 0, cuda_stream>>>(tmp_out, a_h, b_h, _length,
                                                    _a_stride, _b_stride);
            CHECK_CUDA(cudaGetLastError());
            // Use a simple device kernel to cast float to half
            // For now, copy to host, cast, and copy back
            float result_f;
            CHECK_CUDA(cudaMemcpyAsync(&result_f, tmp_out, sizeof(float),
                                       cudaMemcpyDeviceToHost, cuda_stream));
            CHECK_CUDA(cudaStreamSynchronize(cuda_stream));
            __half h_result = __float2half(result_f);
            CHECK_CUDA(cudaMemcpyAsync(out, &h_result, sizeof(__half),
                                       cudaMemcpyHostToDevice, cuda_stream));
        }
        break;
    }
    case INFINI_DTYPE_BF16: {
        // For BF16, accumulate in float, then cast back
        if (workspace_size < sizeof(float)) {
            return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
        }
        float *tmp_out = reinterpret_cast<float *>(workspace);
        {
            const __nv_bfloat16 *a_bf = reinterpret_cast<const __nv_bfloat16 *>(a);
            const __nv_bfloat16 *b_bf = reinterpret_cast<const __nv_bfloat16 *>(b);
            op::vdot::cuda::vdotKernel<BLOCK_SIZE, __nv_bfloat16, float>
                <<<1, BLOCK_SIZE, 0, cuda_stream>>>(tmp_out, a_bf, b_bf, _length,
                                                    _a_stride, _b_stride);
            CHECK_CUDA(cudaGetLastError());
            float result_f;
            CHECK_CUDA(cudaMemcpyAsync(&result_f, tmp_out, sizeof(float),
                                       cudaMemcpyDeviceToHost, cuda_stream));
            CHECK_CUDA(cudaStreamSynchronize(cuda_stream));
            __nv_bfloat16 bf_result = __float2bfloat16(result_f);
            CHECK_CUDA(cudaMemcpyAsync(out, &bf_result, sizeof(__nv_bfloat16),
                                       cudaMemcpyHostToDevice, cuda_stream));
        }
        break;
    }
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::vdot::moore
