#include "interpolate_moore.h"
#include "../cuda/kernel.cuh"
#include "../../../../utils.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstring>

namespace op::interpolate::moore {

static bool try_parse_mode(const char *mode_str, InterpolateMode &mode) {
    if (std::strcmp(mode_str, "nearest") == 0) {
        mode = InterpolateMode::NEAREST;
        return true;
    } else if (std::strcmp(mode_str, "linear") == 0) {
        mode = InterpolateMode::LINEAR;
        return true;
    } else if (std::strcmp(mode_str, "bilinear") == 0) {
        mode = InterpolateMode::BILINEAR;
        return true;
    } else if (std::strcmp(mode_str, "trilinear") == 0) {
        mode = InterpolateMode::TRILINEAR;
        return true;
    } else if (std::strcmp(mode_str, "area") == 0) {
        mode = InterpolateMode::AREA;
        return true;
    }
    return false;
}

static double compute_scale(size_t in_size, size_t out_size, int align_corners) {
    if (out_size == 0) {
        return 0.0;
    }
    if (align_corners) {
        return (out_size > 1) ? (static_cast<double>(in_size) - 1.0) / (static_cast<double>(out_size) - 1.0) : 0.0;
    }
    return static_cast<double>(in_size) / static_cast<double>(out_size);
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

    if ((size != nullptr) == (scale_factor != nullptr)) {
        return INFINI_STATUS_BAD_PARAM;
    }
    if (y_shape.size() != x_shape.size() || y_shape[0] != x_shape[0] || y_shape[1] != x_shape[1]) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    size_t ndim = x_shape.size() - 2;

    if (scale_factor != nullptr) {
        const double *scale_array = reinterpret_cast<const double *>(scale_factor);
        const double scale = scale_array[0];
        std::vector<size_t> expected_y_shape = x_shape;
        for (size_t i = 0; i < ndim; ++i) {
            expected_y_shape[i + 2] = static_cast<size_t>(static_cast<double>(x_shape[i + 2]) * scale);
        }
        if (y_shape != expected_y_shape) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
    }

    InterpolateMode parsed_mode{};
    if (!try_parse_mode(mode, parsed_mode)) {
        return INFINI_STATUS_BAD_PARAM;
    }

    *desc_ptr = new Descriptor(dtype, ndim, x_shape, y_shape, parsed_mode, align_corners,
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

    auto musa_stream = reinterpret_cast<musaStream_t>(stream);
    constexpr int BLOCK_SIZE = 256;
    int num_blocks = (output_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    size_t batch = input_shape[0];
    size_t channels = input_shape[1];

    if (mode == InterpolateMode::BILINEAR && ndim == 2) {
        size_t in_h = input_shape[2];
        size_t in_w = input_shape[3];
        size_t out_h = output_shape[2];
        size_t out_w = output_shape[3];
        double scale_h = compute_scale(in_h, out_h, align_corners);
        double scale_w = compute_scale(in_w, out_w, align_corners);

        switch (_dtype) {
        case INFINI_DTYPE_F16:
            cuda::interpolate_bilinear_2d_kernel<half><<<num_blocks, BLOCK_SIZE, 0, musa_stream>>>(
                reinterpret_cast<half *>(y),
                reinterpret_cast<const half *>(x),
                batch, channels, in_h, in_w, out_h, out_w, scale_h, scale_w, align_corners);
            break;
        case INFINI_DTYPE_BF16:
            cuda::interpolate_bilinear_2d_kernel<cuda_bfloat16><<<num_blocks, BLOCK_SIZE, 0, musa_stream>>>(
                reinterpret_cast<cuda_bfloat16 *>(y),
                reinterpret_cast<const cuda_bfloat16 *>(x),
                batch, channels, in_h, in_w, out_h, out_w, scale_h, scale_w, align_corners);
            break;
        case INFINI_DTYPE_F32:
            cuda::interpolate_bilinear_2d_kernel<float><<<num_blocks, BLOCK_SIZE, 0, musa_stream>>>(
                reinterpret_cast<float *>(y),
                reinterpret_cast<const float *>(x),
                batch, channels, in_h, in_w, out_h, out_w, scale_h, scale_w, align_corners);
            break;
        case INFINI_DTYPE_F64:
            cuda::interpolate_bilinear_2d_kernel<double><<<num_blocks, BLOCK_SIZE, 0, musa_stream>>>(
                reinterpret_cast<double *>(y),
                reinterpret_cast<const double *>(x),
                batch, channels, in_h, in_w, out_h, out_w, scale_h, scale_w, align_corners);
            break;
        default:
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::interpolate::moore
