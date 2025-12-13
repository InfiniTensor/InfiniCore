#include "dist_moore.h"
#include "../cuda/kernel.cuh"
#include "../../../utils.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace op::dist::moore {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x1_desc,
    infiniopTensorDescriptor_t x2_desc,
    double p) {

    auto dtype = x1_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64, INFINI_DTYPE_BF16);

    auto x1_shape = x1_desc->shape();
    auto x2_shape = x2_desc->shape();
    auto y_shape = y_desc->shape();

    if (x1_shape != x2_shape) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    if (y_shape.size() != 0 && (y_shape.size() != 1 || y_shape[0] != 1)) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    size_t input_size = x1_desc->numel();
    ptrdiff_t x1_stride = (x1_desc->isContiguous()) ? 1 : x1_desc->strides()[x1_desc->ndim() - 1];
    ptrdiff_t x2_stride = (x2_desc->isContiguous()) ? 1 : x2_desc->strides()[x2_desc->ndim() - 1];

    *desc_ptr = new Descriptor(dtype, input_size, p, x1_stride, x2_stride,
                               handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x1,
    const void *x2,
    void *stream) const {

    auto musa_stream = reinterpret_cast<musaStream_t>(stream);
    constexpr int BLOCK_SIZE = 256;

    switch (_dtype) {
    case INFINI_DTYPE_F16: {
        float *result_f = nullptr;
        CHECK_MOORE(musaMalloc((void **)&result_f, sizeof(float)));
        CHECK_MOORE(musaMemsetAsync(result_f, 0, sizeof(float), musa_stream));
        cuda::dist_kernel<BLOCK_SIZE, half, float><<<1, BLOCK_SIZE, 0, musa_stream>>>(
            result_f, reinterpret_cast<const half *>(x1), reinterpret_cast<const half *>(x2),
            _input_size, _p, _x1_stride, _x2_stride);
        float result_val;
        CHECK_MOORE(musaMemcpyAsync(&result_val, result_f, sizeof(float), musaMemcpyDeviceToHost, musa_stream));
        CHECK_MOORE(musaStreamSynchronize(musa_stream));
        *reinterpret_cast<half *>(y) = __float2half(result_val);
        CHECK_MOORE(musaFree(result_f));
        break;
    }
    case INFINI_DTYPE_BF16: {
        float *result_f = nullptr;
        CHECK_MOORE(musaMalloc((void **)&result_f, sizeof(float)));
        CHECK_MOORE(musaMemsetAsync(result_f, 0, sizeof(float), musa_stream));
        cuda::dist_kernel<BLOCK_SIZE, cuda_bfloat16, float><<<1, BLOCK_SIZE, 0, musa_stream>>>(
            result_f, reinterpret_cast<const cuda_bfloat16 *>(x1), reinterpret_cast<const cuda_bfloat16 *>(x2),
            _input_size, _p, _x1_stride, _x2_stride);
        float result_val;
        CHECK_MOORE(musaMemcpyAsync(&result_val, result_f, sizeof(float), musaMemcpyDeviceToHost, musa_stream));
        CHECK_MOORE(musaStreamSynchronize(musa_stream));
        *reinterpret_cast<cuda_bfloat16 *>(y) = __float2bfloat16_rn(result_val);
        CHECK_MOORE(musaFree(result_f));
        break;
    }
    case INFINI_DTYPE_F32: {
        float *result_f = reinterpret_cast<float *>(y);
        CHECK_MOORE(musaMemsetAsync(result_f, 0, sizeof(float), musa_stream));
        cuda::dist_kernel<BLOCK_SIZE, float, float><<<1, BLOCK_SIZE, 0, musa_stream>>>(
            result_f, reinterpret_cast<const float *>(x1), reinterpret_cast<const float *>(x2),
            _input_size, _p, _x1_stride, _x2_stride);
        break;
    }
    case INFINI_DTYPE_F64: {
        double *result_d = reinterpret_cast<double *>(y);
        CHECK_MOORE(musaMemsetAsync(result_d, 0, sizeof(double), musa_stream));
        cuda::dist_kernel<BLOCK_SIZE, double, double><<<1, BLOCK_SIZE, 0, musa_stream>>>(
            result_d, reinterpret_cast<const double *>(x1), reinterpret_cast<const double *>(x2),
            _input_size, _p, _x1_stride, _x2_stride);
        break;
    }
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::dist::moore
