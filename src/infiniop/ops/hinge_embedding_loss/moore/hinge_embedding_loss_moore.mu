#include "hinge_embedding_loss_moore.h"
#include "../cuda/kernel.cuh"
#include "../../../../utils.h"
#include "../../../tensor.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <limits>

namespace op::hinge_embedding_loss::moore {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t target_desc,
    double margin,
    int reduction) {

    auto dtype = input_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64, INFINI_DTYPE_BF16);
    if (target_desc->dtype() != dtype || y_desc->dtype() != dtype) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    auto input_shape = input_desc->shape();
    auto target_shape = target_desc->shape();
    auto y_shape = y_desc->shape();

    if (input_shape != target_shape) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    Reduction red = static_cast<Reduction>(reduction);
    if (red != Reduction::NONE && (dtype == INFINI_DTYPE_F16 || dtype == INFINI_DTYPE_BF16)) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    if (!input_desc->isContiguous() || !target_desc->isContiguous() ||
        (red == Reduction::NONE && !y_desc->isContiguous())) {
        return INFINI_STATUS_BAD_TENSOR_STRIDES;
    }
    std::vector<size_t> expected_y_shape;
    if (red == Reduction::NONE) {
        expected_y_shape = input_shape;
    } else {
        expected_y_shape = {};
    }

    if (y_shape != expected_y_shape) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    *desc_ptr = new Descriptor(dtype, input_desc->numel(), margin, red,
                               handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *input,
    const void *target,
    void *stream) const {

    auto musa_stream = reinterpret_cast<musaStream_t>(stream);
    constexpr int BLOCK_SIZE = 256;
    if (input_size == 0) {
        if (reduction != Reduction::NONE) {
            if (_dtype == INFINI_DTYPE_F32) {
                float value = (reduction == Reduction::MEAN) ? std::numeric_limits<float>::quiet_NaN() : 0.0f;
                CHECK_MOORE(musaMemcpyAsync(y, &value, sizeof(value), musaMemcpyHostToDevice, musa_stream));
            } else if (_dtype == INFINI_DTYPE_F64) {
                double value = (reduction == Reduction::MEAN) ? std::numeric_limits<double>::quiet_NaN() : 0.0;
                CHECK_MOORE(musaMemcpyAsync(y, &value, sizeof(value), musaMemcpyHostToDevice, musa_stream));
            }
        }
        return INFINI_STATUS_SUCCESS;
    }
    int num_blocks = (input_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if (reduction == Reduction::NONE) {
        switch (_dtype) {
        case INFINI_DTYPE_F16: {
            float margin_val = static_cast<float>(margin);
            cuda::hinge_embedding_loss_none_kernel<half, float><<<num_blocks, BLOCK_SIZE, 0, musa_stream>>>(
                reinterpret_cast<half *>(y),
                reinterpret_cast<const half *>(input),
                reinterpret_cast<const half *>(target),
                input_size,
                /*ndim=*/0,
                /*shape=*/nullptr,
                /*output_strides=*/nullptr,
                /*input_strides=*/nullptr,
                /*target_strides=*/nullptr,
                /*output_contiguous=*/true,
                /*input_contiguous=*/true,
                /*target_contiguous=*/true,
                margin_val);
            break;
        }
        case INFINI_DTYPE_BF16: {
            float margin_val = static_cast<float>(margin);
            cuda::hinge_embedding_loss_none_kernel<cuda_bfloat16, float><<<num_blocks, BLOCK_SIZE, 0, musa_stream>>>(
                reinterpret_cast<cuda_bfloat16 *>(y),
                reinterpret_cast<const cuda_bfloat16 *>(input),
                reinterpret_cast<const cuda_bfloat16 *>(target),
                input_size,
                /*ndim=*/0,
                /*shape=*/nullptr,
                /*output_strides=*/nullptr,
                /*input_strides=*/nullptr,
                /*target_strides=*/nullptr,
                /*output_contiguous=*/true,
                /*input_contiguous=*/true,
                /*target_contiguous=*/true,
                margin_val);
            break;
        }
        case INFINI_DTYPE_F32: {
            float margin_val = static_cast<float>(margin);
            cuda::hinge_embedding_loss_none_kernel<float, float><<<num_blocks, BLOCK_SIZE, 0, musa_stream>>>(
                reinterpret_cast<float *>(y),
                reinterpret_cast<const float *>(input),
                reinterpret_cast<const float *>(target),
                input_size,
                /*ndim=*/0,
                /*shape=*/nullptr,
                /*output_strides=*/nullptr,
                /*input_strides=*/nullptr,
                /*target_strides=*/nullptr,
                /*output_contiguous=*/true,
                /*input_contiguous=*/true,
                /*target_contiguous=*/true,
                margin_val);
            break;
        }
        case INFINI_DTYPE_F64: {
            double margin_val = margin;
            cuda::hinge_embedding_loss_none_kernel<double, double><<<num_blocks, BLOCK_SIZE, 0, musa_stream>>>(
                reinterpret_cast<double *>(y),
                reinterpret_cast<const double *>(input),
                reinterpret_cast<const double *>(target),
                input_size,
                /*ndim=*/0,
                /*shape=*/nullptr,
                /*output_strides=*/nullptr,
                /*input_strides=*/nullptr,
                /*target_strides=*/nullptr,
                /*output_contiguous=*/true,
                /*input_contiguous=*/true,
                /*target_contiguous=*/true,
                margin_val);
            break;
        }
        default:
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
    } else {
        // Sum or Mean reduction
        const bool mean = (reduction == Reduction::MEAN);
        switch (_dtype) {
        case INFINI_DTYPE_F32: {
            float margin_val = static_cast<float>(margin);
            CHECK_MOORE(musaMemsetAsync(y, 0, sizeof(float), musa_stream));
            cuda::hinge_embedding_loss_reduce_kernel<float, float, BLOCK_SIZE><<<num_blocks, BLOCK_SIZE, 0, musa_stream>>>(
                reinterpret_cast<float *>(y),
                reinterpret_cast<const float *>(input),
                reinterpret_cast<const float *>(target),
                input_size,
                /*ndim=*/0,
                /*shape=*/nullptr,
                /*input_strides=*/nullptr,
                /*target_strides=*/nullptr,
                /*input_contiguous=*/true,
                /*target_contiguous=*/true,
                margin_val);
            cuda::hinge_embedding_loss_finalize_kernel<float, float><<<1, 1, 0, musa_stream>>>(
                reinterpret_cast<float *>(y),
                reinterpret_cast<const float *>(y),
                input_size,
                mean);
            break;
        }
        case INFINI_DTYPE_F64: {
            double margin_val = margin;
            CHECK_MOORE(musaMemsetAsync(y, 0, sizeof(double), musa_stream));
            cuda::hinge_embedding_loss_reduce_kernel<double, double, BLOCK_SIZE><<<num_blocks, BLOCK_SIZE, 0, musa_stream>>>(
                reinterpret_cast<double *>(y),
                reinterpret_cast<const double *>(input),
                reinterpret_cast<const double *>(target),
                input_size,
                /*ndim=*/0,
                /*shape=*/nullptr,
                /*input_strides=*/nullptr,
                /*target_strides=*/nullptr,
                /*input_contiguous=*/true,
                /*target_contiguous=*/true,
                margin_val);
            cuda::hinge_embedding_loss_finalize_kernel<double, double><<<1, 1, 0, musa_stream>>>(
                reinterpret_cast<double *>(y),
                reinterpret_cast<const double *>(y),
                input_size,
                mean);
            break;
        }
        default:
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::hinge_embedding_loss::moore
