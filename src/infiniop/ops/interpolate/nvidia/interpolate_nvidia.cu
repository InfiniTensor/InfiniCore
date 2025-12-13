#include "interpolate_nvidia.cuh"
#include "../cuda/kernel.cuh"
#include "../../../utils.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstring>

namespace op::interpolate::nvidia {

InterpolateMode parseMode(const char *mode_str) {
    if (std::strcmp(mode_str, "nearest") == 0) {
        return InterpolateMode::NEAREST;
    } else if (std::strcmp(mode_str, "linear") == 0) {
        return InterpolateMode::LINEAR;
    } else if (std::strcmp(mode_str, "bilinear") == 0) {
        return InterpolateMode::BILINEAR;
    } else if (std::strcmp(mode_str, "trilinear") == 0) {
        return InterpolateMode::TRILINEAR;
    } else if (std::strcmp(mode_str, "area") == 0) {
        return InterpolateMode::AREA;
    }
    return InterpolateMode::NEAREST;
}

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    const char *mode,
    void *size,
    void *scale_factor,
    int align_corners) {

    auto dtype = x_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64, INFINI_DTYPE_BF16);

    auto x_shape = x_desc->shape();
    auto y_shape = y_desc->shape();

    if (x_shape.size() < 3) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    size_t ndim = x_shape.size() - 2;

    std::vector<size_t> expected_y_shape = x_shape;
    if (size != nullptr) {
        const int64_t *size_array = reinterpret_cast<const int64_t *>(size);
        for (size_t i = 0; i < ndim; ++i) {
            expected_y_shape[i + 2] = static_cast<size_t>(size_array[i]);
        }
    } else if (scale_factor != nullptr) {
        const double *scale_array = reinterpret_cast<const double *>(scale_factor);
        if (ndim == 1) {
            double scale = scale_array[0];
            expected_y_shape[2] = static_cast<size_t>(x_shape[2] * scale);
        } else {
            for (size_t i = 0; i < ndim; ++i) {
                double scale = scale_array[i];
                expected_y_shape[i + 2] = static_cast<size_t>(x_shape[i + 2] * scale);
            }
        }
    } else {
        return INFINI_STATUS_BAD_PARAM;
    }

    if (y_shape != expected_y_shape) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    *desc_ptr = new Descriptor(dtype, ndim, x_shape, y_shape, parseMode(mode), align_corners,
                               x_desc->numel(), y_desc->numel(),
                               handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    void *stream) const {

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    constexpr int BLOCK_SIZE = 256;
    int num_blocks = (output_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    size_t batch = input_shape[0];
    size_t channels = input_shape[1];

    if (mode == InterpolateMode::BILINEAR && ndim == 2) {
        size_t in_h = input_shape[2];
        size_t in_w = input_shape[3];
        size_t out_h = output_shape[2];
        size_t out_w = output_shape[3];
        double scale_h = align_corners ? (in_h - 1.0) / (out_h - 1.0) : static_cast<double>(in_h) / out_h;
        double scale_w = align_corners ? (in_w - 1.0) / (out_w - 1.0) : static_cast<double>(in_w) / out_w;

        switch (_dtype) {
        case INFINI_DTYPE_F16:
            cuda::interpolate_bilinear_2d_kernel<half><<<num_blocks, BLOCK_SIZE, 0, cuda_stream>>>(
                reinterpret_cast<half *>(y),
                reinterpret_cast<const half *>(x),
                batch, channels, in_h, in_w, out_h, out_w, scale_h, scale_w, align_corners);
            break;
        case INFINI_DTYPE_BF16:
            cuda::interpolate_bilinear_2d_kernel<cuda_bfloat16><<<num_blocks, BLOCK_SIZE, 0, cuda_stream>>>(
                reinterpret_cast<cuda_bfloat16 *>(y),
                reinterpret_cast<const cuda_bfloat16 *>(x),
                batch, channels, in_h, in_w, out_h, out_w, scale_h, scale_w, align_corners);
            break;
        case INFINI_DTYPE_F32:
            cuda::interpolate_bilinear_2d_kernel<float><<<num_blocks, BLOCK_SIZE, 0, cuda_stream>>>(
                reinterpret_cast<float *>(y),
                reinterpret_cast<const float *>(x),
                batch, channels, in_h, in_w, out_h, out_w, scale_h, scale_w, align_corners);
            break;
        case INFINI_DTYPE_F64:
            cuda::interpolate_bilinear_2d_kernel<double><<<num_blocks, BLOCK_SIZE, 0, cuda_stream>>>(
                reinterpret_cast<double *>(y),
                reinterpret_cast<const double *>(x),
                batch, channels, in_h, in_w, out_h, out_w, scale_h, scale_w, align_corners);
            break;
        default:
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
    } else {
        // For other modes, use CPU fallback for now
        // TODO: Implement full GPU kernels for all modes
        // Copy to CPU, compute, copy back
        size_t input_bytes = input_size * infiniopGetDtypeSize(_dtype);
        size_t output_bytes = output_size * infiniopGetDtypeSize(_dtype);
        
        // Allocate host memory
        std::vector<uint8_t> h_input(input_bytes);
        std::vector<uint8_t> h_output(output_bytes);
        
        CHECK_CUDA(cudaMemcpyAsync(h_input.data(), x, input_bytes, cudaMemcpyDeviceToHost, cuda_stream));
        CHECK_CUDA(cudaStreamSynchronize(cuda_stream));
        
        // Call CPU implementation
        // Create temporary CPU descriptors and call CPU implementation
        // For now, return not implemented
        return INFINI_STATUS_NOT_IMPLEMENTED;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::interpolate::nvidia
