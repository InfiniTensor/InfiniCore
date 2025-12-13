#include "hinge_embedding_loss_nvidia.cuh"
#include "../cuda/kernel.cuh"
#include "../../../utils.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace op::hinge_embedding_loss::nvidia {

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

    auto input_shape = input_desc->shape();
    auto target_shape = target_desc->shape();
    auto y_shape = y_desc->shape();

    if (input_shape != target_shape) {
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

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    constexpr int BLOCK_SIZE = 256;
    int num_blocks = (input_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if (reduction == Reduction::NONE) {
        // Element-wise loss
        switch (_dtype) {
        case INFINI_DTYPE_F16: {
            half margin_val = __float2half(static_cast<float>(margin));
            cuda::hinge_embedding_loss_kernel<half><<<num_blocks, BLOCK_SIZE, 0, cuda_stream>>>(
                reinterpret_cast<half *>(y),
                reinterpret_cast<const half *>(input),
                reinterpret_cast<const half *>(target),
                input_size, margin_val);
            break;
        }
        case INFINI_DTYPE_BF16: {
            cuda_bfloat16 margin_val = __float2bfloat16_rn(static_cast<float>(margin));
            cuda::hinge_embedding_loss_kernel<cuda_bfloat16><<<num_blocks, BLOCK_SIZE, 0, cuda_stream>>>(
                reinterpret_cast<cuda_bfloat16 *>(y),
                reinterpret_cast<const cuda_bfloat16 *>(input),
                reinterpret_cast<const cuda_bfloat16 *>(target),
                input_size, margin_val);
            break;
        }
        case INFINI_DTYPE_F32: {
            float margin_val = static_cast<float>(margin);
            cuda::hinge_embedding_loss_kernel<float><<<num_blocks, BLOCK_SIZE, 0, cuda_stream>>>(
                reinterpret_cast<float *>(y),
                reinterpret_cast<const float *>(input),
                reinterpret_cast<const float *>(target),
                input_size, margin_val);
            break;
        }
        case INFINI_DTYPE_F64: {
            double margin_val = margin;
            cuda::hinge_embedding_loss_kernel<double><<<num_blocks, BLOCK_SIZE, 0, cuda_stream>>>(
                reinterpret_cast<double *>(y),
                reinterpret_cast<const double *>(input),
                reinterpret_cast<const double *>(target),
                input_size, margin_val);
            break;
        }
        default:
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
    } else {
        // Sum or Mean: use reduction kernel
        switch (_dtype) {
        case INFINI_DTYPE_F16: {
            float margin_val = static_cast<float>(margin);
            float *result_f = nullptr;
            CHECK_CUDA(cudaMallocAsync(&result_f, sizeof(float), cuda_stream));
            CHECK_CUDA(cudaMemsetAsync(result_f, 0, sizeof(float), cuda_stream));
            cuda::hinge_embedding_loss_reduce_kernel<half, float><<<num_blocks, BLOCK_SIZE, 0, cuda_stream>>>(
                reinterpret_cast<half *>(y),
                reinterpret_cast<const half *>(input),
                reinterpret_cast<const half *>(target),
                input_size, margin_val);
            if (reduction == Reduction::MEAN) {
                // Divide by input_size
                float scale = 1.0f / static_cast<float>(input_size);
                // TODO: Add scaling kernel
            }
            float result_val;
            CHECK_CUDA(cudaMemcpyAsync(&result_val, result_f, sizeof(float), cudaMemcpyDeviceToHost, cuda_stream));
            CHECK_CUDA(cudaStreamSynchronize(cuda_stream));
            *reinterpret_cast<half *>(y) = __float2half(result_val);
            CHECK_CUDA(cudaFreeAsync(result_f, cuda_stream));
            break;
        }
        case INFINI_DTYPE_F32: {
            float margin_val = static_cast<float>(margin);
            CHECK_CUDA(cudaMemsetAsync(y, 0, sizeof(float), cuda_stream));
            cuda::hinge_embedding_loss_reduce_kernel<float, float><<<num_blocks, BLOCK_SIZE, 0, cuda_stream>>>(
                reinterpret_cast<float *>(y),
                reinterpret_cast<const float *>(input),
                reinterpret_cast<const float *>(target),
                input_size, margin_val);
            if (reduction == Reduction::MEAN) {
                // Scale by 1/n
                // TODO: Add scaling
            }
            break;
        }
        case INFINI_DTYPE_F64: {
            double margin_val = margin;
            CHECK_CUDA(cudaMemsetAsync(y, 0, sizeof(double), cuda_stream));
            cuda::hinge_embedding_loss_reduce_kernel<double, double><<<num_blocks, BLOCK_SIZE, 0, cuda_stream>>>(
                reinterpret_cast<double *>(y),
                reinterpret_cast<const double *>(input),
                reinterpret_cast<const double *>(target),
                input_size, margin_val);
            break;
        }
        default:
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::hinge_embedding_loss::nvidia
