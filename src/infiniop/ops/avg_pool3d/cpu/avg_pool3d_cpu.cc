#include "avg_pool3d_cpu.h"
#include "../../../utils.h"
#include <algorithm>
#include <cmath>

namespace op::avg_pool3d::cpu {

utils::Result<AvgPool3dInfo> AvgPool3dInfo::create(
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t y_desc,
    void *kernel_size,
    void *stride,
    void *padding) {

    auto x_shape = x_desc->shape();
    auto y_shape = y_desc->shape();

    if (x_shape.size() != 5 || y_shape.size() != 5) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    size_t batch = x_shape[0];
    size_t channels = x_shape[1];
    size_t input_d = x_shape[2];
    size_t input_h = x_shape[3];
    size_t input_w = x_shape[4];

    // Parse kernel_size
    size_t kernel_d, kernel_h, kernel_w;
    if (kernel_size) {
        size_t *ks = reinterpret_cast<size_t *>(kernel_size);
        if (ks[0] == 0 || ks[1] == 0 || ks[2] == 0) {
            return INFINI_STATUS_BAD_PARAM;
        }
        kernel_d = ks[0];
        kernel_h = ks[1];
        kernel_w = ks[2];
    } else {
        return INFINI_STATUS_BAD_PARAM;
    }

    // Parse stride (default to kernel_size if not provided)
    size_t stride_d, stride_h, stride_w;
    if (stride) {
        size_t *s = reinterpret_cast<size_t *>(stride);
        stride_d = s[0];
        stride_h = s[1];
        stride_w = s[2];
    } else {
        stride_d = kernel_d;
        stride_h = kernel_h;
        stride_w = kernel_w;
    }

    // Parse padding
    size_t pad_d, pad_h, pad_w;
    if (padding) {
        size_t *p = reinterpret_cast<size_t *>(padding);
        // Assume it's always a tuple of 3 values for 3D pooling
        pad_d = p[0];
        pad_h = p[1];
        pad_w = p[2];
    } else {
        pad_d = pad_h = pad_w = 0;
    }

    // Calculate output dimensions
    size_t output_d = (input_d + 2 * pad_d - kernel_d) / stride_d + 1;
    size_t output_h = (input_h + 2 * pad_h - kernel_h) / stride_h + 1;
    size_t output_w = (input_w + 2 * pad_w - kernel_w) / stride_w + 1;

    // Verify output shape
    if (y_shape[0] != batch || y_shape[1] != channels ||
        y_shape[2] != output_d || y_shape[3] != output_h || y_shape[4] != output_w) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    AvgPool3dInfo info;
    info.batch = batch;
    info.channels = channels;
    info.input_d = input_d;
    info.input_h = input_h;
    info.input_w = input_w;
    info.output_d = output_d;
    info.output_h = output_h;
    info.output_w = output_w;
    info.kernel_d = kernel_d;
    info.kernel_h = kernel_h;
    info.kernel_w = kernel_w;
    info.stride_d = stride_d;
    info.stride_h = stride_h;
    info.stride_w = stride_w;
    info.pad_d = pad_d;
    info.pad_h = pad_h;
    info.pad_w = pad_w;
    info.input_strides = x_desc->strides();
    info.output_strides = y_desc->strides();

    return utils::Result<AvgPool3dInfo>(std::move(info));
}

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    void *kernel_size,
    void *stride,
    void *padding) {

    auto dtype = x_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64, INFINI_DTYPE_BF16);

    auto info_result = AvgPool3dInfo::create(x_desc, y_desc, kernel_size, stride, padding);
    CHECK_RESULT(info_result);

    *desc_ptr = new Descriptor(dtype, info_result.take(), handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <typename T>
void avg_pool3d_impl(
    const AvgPool3dInfo &info,
    T *y,
    const T *x) {

    const size_t kernel_size = info.kernel_d * info.kernel_h * info.kernel_w;
    const float inv_kernel_size = 1.0f / static_cast<float>(kernel_size);

#pragma omp parallel for collapse(2)
    for (ptrdiff_t b = 0; b < static_cast<ptrdiff_t>(info.batch); ++b) {
        for (ptrdiff_t c = 0; c < static_cast<ptrdiff_t>(info.channels); ++c) {
            for (size_t od = 0; od < info.output_d; ++od) {
                for (size_t oh = 0; oh < info.output_h; ++oh) {
                    for (size_t ow = 0; ow < info.output_w; ++ow) {
                        float sum = 0.0f;
                        size_t count = 0;

                        // Calculate input window
                        size_t id_start = od * info.stride_d - info.pad_d;
                        size_t ih_start = oh * info.stride_h - info.pad_h;
                        size_t iw_start = ow * info.stride_w - info.pad_w;

                        for (size_t kd = 0; kd < info.kernel_d; ++kd) {
                            for (size_t kh = 0; kh < info.kernel_h; ++kh) {
                                for (size_t kw = 0; kw < info.kernel_w; ++kw) {
                                    ptrdiff_t id = static_cast<ptrdiff_t>(id_start + kd);
                                    ptrdiff_t ih = static_cast<ptrdiff_t>(ih_start + kh);
                                    ptrdiff_t iw = static_cast<ptrdiff_t>(iw_start + kw);

                                    // Check bounds (accounting for padding)
                                    if (id >= 0 && id < static_cast<ptrdiff_t>(info.input_d) &&
                                        ih >= 0 && ih < static_cast<ptrdiff_t>(info.input_h) &&
                                        iw >= 0 && iw < static_cast<ptrdiff_t>(info.input_w)) {
                                        size_t x_idx = b * info.input_strides[0] +
                                                      c * info.input_strides[1] +
                                                      static_cast<size_t>(id) * info.input_strides[2] +
                                                      static_cast<size_t>(ih) * info.input_strides[3] +
                                                      static_cast<size_t>(iw) * info.input_strides[4];
                                        sum += utils::cast<float>(x[x_idx]);
                                        count++;
                                    }
                                }
                            }
                        }

                        size_t y_idx = b * info.output_strides[0] +
                                      c * info.output_strides[1] +
                                      od * info.output_strides[2] +
                                      oh * info.output_strides[3] +
                                      ow * info.output_strides[4];
                        y[y_idx] = utils::cast<T>(sum / static_cast<float>(count > 0 ? count : 1));
                    }
                }
            }
        }
    }
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    void *stream) const {

    switch (_dtype) {
    case INFINI_DTYPE_F16:
        avg_pool3d_impl<fp16_t>(_info, reinterpret_cast<fp16_t *>(y), reinterpret_cast<const fp16_t *>(x));
        break;
    case INFINI_DTYPE_BF16:
        avg_pool3d_impl<bf16_t>(_info, reinterpret_cast<bf16_t *>(y), reinterpret_cast<const bf16_t *>(x));
        break;
    case INFINI_DTYPE_F32:
        avg_pool3d_impl<float>(_info, reinterpret_cast<float *>(y), reinterpret_cast<const float *>(x));
        break;
    case INFINI_DTYPE_F64:
        avg_pool3d_impl<double>(_info, reinterpret_cast<double *>(y), reinterpret_cast<const double *>(x));
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::avg_pool3d::cpu
