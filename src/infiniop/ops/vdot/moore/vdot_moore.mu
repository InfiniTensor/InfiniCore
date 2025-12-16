#include "../../../devices/moore/moore_common.h"
#include "vdot_moore.h"
#include "../../../devices/moore/moore_kernel_common.h"
#include "../cuda/kernel.cuh"

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

    auto musa_stream = reinterpret_cast<musaStream_t>(stream);
    constexpr unsigned int BLOCK_SIZE = 256;

    switch (_in_dtype) {
    case INFINI_DTYPE_F32: {
        float *out_f = reinterpret_cast<float *>(out);
        const float *a_f = reinterpret_cast<const float *>(a);
        const float *b_f = reinterpret_cast<const float *>(b);
        op::vdot::cuda::vdotKernel<BLOCK_SIZE, float, float>
            <<<1, BLOCK_SIZE, 0, musa_stream>>>(out_f, a_f, b_f, _length, _a_stride,
                                                _b_stride);
        break;
    }
    case INFINI_DTYPE_F64: {
        double *out_d = reinterpret_cast<double *>(out);
        const double *a_d = reinterpret_cast<const double *>(a);
        const double *b_d = reinterpret_cast<const double *>(b);
        op::vdot::cuda::vdotKernel<BLOCK_SIZE, double, double>
            <<<1, BLOCK_SIZE, 0, musa_stream>>>(out_d, a_d, b_d, _length, _a_stride,
                                                _b_stride);
        break;
    }
    case INFINI_DTYPE_F16: {
        // For FP16, accumulate in float, then cast back to half
        if (workspace_size < sizeof(float)) {
            return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
        }
        float *tmp_out = reinterpret_cast<float *>(workspace);
        {
            const __half *a_h = reinterpret_cast<const __half *>(a);
            const __half *b_h = reinterpret_cast<const __half *>(b);
            op::vdot::cuda::vdotKernel<BLOCK_SIZE, __half, float>
                <<<1, BLOCK_SIZE, 0, musa_stream>>>(tmp_out, a_h, b_h, _length,
                                                    _a_stride, _b_stride);
            float result_f;
            CHECK_MOORE(musaMemcpyAsync(&result_f, tmp_out, sizeof(float),
                                       musaMemcpyDeviceToHost, musa_stream));
            CHECK_MOORE(musaStreamSynchronize(musa_stream));
            __half h_result = __float2half(result_f);
            CHECK_MOORE(musaMemcpyAsync(out, &h_result, sizeof(__half),
                                       musaMemcpyHostToDevice, musa_stream));
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
            const __mt_bfloat16 *a_bf = reinterpret_cast<const __mt_bfloat16 *>(a);
            const __mt_bfloat16 *b_bf = reinterpret_cast<const __mt_bfloat16 *>(b);
            op::vdot::cuda::vdotKernel<BLOCK_SIZE, __mt_bfloat16, float>
                <<<1, BLOCK_SIZE, 0, musa_stream>>>(tmp_out, a_bf, b_bf, _length,
                                                    _a_stride, _b_stride);
            float result_f;
            CHECK_MOORE(musaMemcpyAsync(&result_f, tmp_out, sizeof(float),
                                       musaMemcpyDeviceToHost, musa_stream));
            CHECK_MOORE(musaStreamSynchronize(musa_stream));
            __mt_bfloat16 bf_result = __float2bfloat16(result_f);
            CHECK_MOORE(musaMemcpyAsync(out, &bf_result, sizeof(__mt_bfloat16),
                                       musaMemcpyHostToDevice, musa_stream));
        }
        break;
    }
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::vdot::moore

