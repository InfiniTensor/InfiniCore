#include "gaussian_nll_loss_moore.h"
#include "../cuda/kernel.cuh"
#include "../../../utils.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace op::gaussian_nll_loss::moore {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t target_desc,
    infiniopTensorDescriptor_t var_desc,
    int full,
    double eps,
    int reduction) {

    auto dtype = input_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64, INFINI_DTYPE_BF16);

    auto input_shape = input_desc->shape();
    auto target_shape = target_desc->shape();
    auto var_shape = var_desc->shape();
    auto y_shape = y_desc->shape();

    if (input_shape != target_shape || input_shape != var_shape) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    Reduction red = static_cast<Reduction>(reduction);
    std::vector<size_t> expected_y_shape;
    if (red == Reduction::NONE) {
        expected_y_shape = input_shape;
    } else {
        expected_y_shape = {};
    }

    if (y_shape != expected_y_shape) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    *desc_ptr = new Descriptor(dtype, input_desc->numel(), full, eps, red,
                               handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *input,
    const void *target,
    const void *var,
    void *stream) const {

    auto musa_stream = reinterpret_cast<musaStream_t>(stream);
    constexpr int BLOCK_SIZE = 256;
    int num_blocks = (input_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if (reduction == Reduction::NONE) {
        switch (_dtype) {
        case INFINI_DTYPE_F16: {
            half eps_val = __float2half(static_cast<float>(eps));
            cuda::gaussian_nll_loss_kernel<half><<<num_blocks, BLOCK_SIZE, 0, musa_stream>>>(
                reinterpret_cast<half *>(y),
                reinterpret_cast<const half *>(input),
                reinterpret_cast<const half *>(target),
                reinterpret_cast<const half *>(var),
                input_size, eps_val, full);
            break;
        }
        case INFINI_DTYPE_BF16: {
            cuda_bfloat16 eps_val = __float2bfloat16_rn(static_cast<float>(eps));
            cuda::gaussian_nll_loss_kernel<cuda_bfloat16><<<num_blocks, BLOCK_SIZE, 0, musa_stream>>>(
                reinterpret_cast<cuda_bfloat16 *>(y),
                reinterpret_cast<const cuda_bfloat16 *>(input),
                reinterpret_cast<const cuda_bfloat16 *>(target),
                reinterpret_cast<const cuda_bfloat16 *>(var),
                input_size, eps_val, full);
            break;
        }
        case INFINI_DTYPE_F32: {
            float eps_val = static_cast<float>(eps);
            cuda::gaussian_nll_loss_kernel<float><<<num_blocks, BLOCK_SIZE, 0, musa_stream>>>(
                reinterpret_cast<float *>(y),
                reinterpret_cast<const float *>(input),
                reinterpret_cast<const float *>(target),
                reinterpret_cast<const float *>(var),
                input_size, eps_val, full);
            break;
        }
        case INFINI_DTYPE_F64: {
            double eps_val = eps;
            cuda::gaussian_nll_loss_kernel<double><<<num_blocks, BLOCK_SIZE, 0, musa_stream>>>(
                reinterpret_cast<double *>(y),
                reinterpret_cast<const double *>(input),
                reinterpret_cast<const double *>(target),
                reinterpret_cast<const double *>(var),
                input_size, eps_val, full);
            break;
        }
        default:
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
    } else {
        switch (_dtype) {
        case INFINI_DTYPE_F32: {
            float eps_val = static_cast<float>(eps);
            CHECK_MOORE(musaMemsetAsync(y, 0, sizeof(float), musa_stream));
            cuda::gaussian_nll_loss_reduce_kernel<float, float><<<num_blocks, BLOCK_SIZE, 0, musa_stream>>>(
                reinterpret_cast<float *>(y),
                reinterpret_cast<const float *>(input),
                reinterpret_cast<const float *>(target),
                reinterpret_cast<const float *>(var),
                input_size, eps_val, full);
            break;
        }
        case INFINI_DTYPE_F64: {
            double eps_val = eps;
            CHECK_MOORE(musaMemsetAsync(y, 0, sizeof(double), musa_stream));
            cuda::gaussian_nll_loss_reduce_kernel<double, double><<<num_blocks, BLOCK_SIZE, 0, musa_stream>>>(
                reinterpret_cast<double *>(y),
                reinterpret_cast<const double *>(input),
                reinterpret_cast<const double *>(target),
                reinterpret_cast<const double *>(var),
                input_size, eps_val, full);
            break;
        }
        default:
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::gaussian_nll_loss::moore
