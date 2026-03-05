#include "interpolate_cpu.h"
#include "../../../utils.h"
#include <cstring>
#include <cmath>
#include <algorithm>

namespace op::interpolate::cpu {

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
    return InterpolateMode::NEAREST;  // Default
}

utils::Result<InterpolateInfo> InterpolateInfo::create(
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t y_desc,
    const char *mode_str,
    void *size,
    void *scale_factor,
    int align_corners) {

    auto x_shape = x_desc->shape();
    auto y_shape = y_desc->shape();

    if (x_shape.size() < 3) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    size_t ndim = x_shape.size() - 2;  // Exclude batch and channel dimensions

    // Calculate output shape
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

    InterpolateInfo info;
    info.ndim = ndim;
    info.input_shape = x_shape;
    info.output_shape = y_shape;
    info.mode = parseMode(mode_str);
    info.align_corners = align_corners;
    info.input_size = x_desc->numel();
    info.output_size = y_desc->numel();

    return utils::Result<InterpolateInfo>(std::move(info));
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

    auto info_result = InterpolateInfo::create(x_desc, y_desc, mode, size, scale_factor, align_corners);
    CHECK_RESULT(info_result);

    *desc_ptr = new Descriptor(dtype, info_result.take(), handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <typename T>
void interpolate_nearest_1d(
    const T *input, T *output,
    size_t batch, size_t channels,
    size_t in_w, size_t out_w,
    int align_corners) {

    double scale = align_corners ? (in_w - 1.0) / (out_w - 1.0) : static_cast<double>(in_w) / out_w;

    for (size_t b = 0; b < batch; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            for (size_t ow = 0; ow < out_w; ++ow) {
                double src_x = align_corners ? ow * scale : (ow + 0.5) * scale - 0.5;
                size_t ix = std::min(static_cast<size_t>(std::round(src_x)), in_w - 1);
                size_t in_idx = (b * channels + c) * in_w + ix;
                size_t out_idx = (b * channels + c) * out_w + ow;
                output[out_idx] = input[in_idx];
            }
        }
    }
}

template <typename T>
void interpolate_bilinear_2d(
    const T *input, T *output,
    size_t batch, size_t channels,
    size_t in_h, size_t in_w,
    size_t out_h, size_t out_w,
    int align_corners) {

    double scale_h = align_corners ? (in_h - 1.0) / (out_h - 1.0) : static_cast<double>(in_h) / out_h;
    double scale_w = align_corners ? (in_w - 1.0) / (out_w - 1.0) : static_cast<double>(in_w) / out_w;

    for (size_t b = 0; b < batch; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            for (size_t oh = 0; oh < out_h; ++oh) {
                for (size_t ow = 0; ow < out_w; ++ow) {
                    double src_y = align_corners ? oh * scale_h : (oh + 0.5) * scale_h - 0.5;
                    double src_x = align_corners ? ow * scale_w : (ow + 0.5) * scale_w - 0.5;

                    src_y = std::max(0.0, std::min(src_y, static_cast<double>(in_h - 1)));
                    src_x = std::max(0.0, std::min(src_x, static_cast<double>(in_w - 1)));

                    size_t y0 = static_cast<size_t>(std::floor(src_y));
                    size_t y1 = std::min(y0 + 1, in_h - 1);
                    size_t x0 = static_cast<size_t>(std::floor(src_x));
                    size_t x1 = std::min(x0 + 1, in_w - 1);

                    double dy = src_y - y0;
                    double dx = src_x - x0;

                    T v00 = input[(b * channels + c) * in_h * in_w + y0 * in_w + x0];
                    T v01 = input[(b * channels + c) * in_h * in_w + y0 * in_w + x1];
                    T v10 = input[(b * channels + c) * in_h * in_w + y1 * in_w + x0];
                    T v11 = input[(b * channels + c) * in_h * in_w + y1 * in_w + x1];

                    T result = utils::cast<T>((1 - dy) * (1 - dx) * utils::cast<double>(v00) +
                                             (1 - dy) * dx * utils::cast<double>(v01) +
                                             dy * (1 - dx) * utils::cast<double>(v10) +
                                             dy * dx * utils::cast<double>(v11));

                    size_t out_idx = (b * channels + c) * out_h * out_w + oh * out_w + ow;
                    output[out_idx] = result;
                }
            }
        }
    }
}

template <typename T>
void interpolate_impl(
    const InterpolateInfo &info,
    T *y,
    const T *x) {

    size_t batch = info.input_shape[0];
    size_t channels = info.input_shape[1];

    if (info.mode == InterpolateMode::NEAREST) {
        if (info.ndim == 1) {
            interpolate_nearest_1d(x, y, batch, channels,
                                  info.input_shape[2], info.output_shape[2],
                                  info.align_corners);
        } else if (info.ndim == 2) {
            // 2D nearest: use bilinear with nearest rounding
            size_t in_h = info.input_shape[2];
            size_t in_w = info.input_shape[3];
            size_t out_h = info.output_shape[2];
            size_t out_w = info.output_shape[3];
            double scale_h = info.align_corners ? (in_h - 1.0) / (out_h - 1.0) : static_cast<double>(in_h) / out_h;
            double scale_w = info.align_corners ? (in_w - 1.0) / (out_w - 1.0) : static_cast<double>(in_w) / out_w;

            for (size_t b = 0; b < batch; ++b) {
                for (size_t c = 0; c < channels; ++c) {
                    for (size_t oh = 0; oh < out_h; ++oh) {
                        for (size_t ow = 0; ow < out_w; ++ow) {
                            double src_y = info.align_corners ? oh * scale_h : (oh + 0.5) * scale_h - 0.5;
                            double src_x = info.align_corners ? ow * scale_w : (ow + 0.5) * scale_w - 0.5;
                            size_t iy = std::min(static_cast<size_t>(std::round(src_y)), in_h - 1);
                            size_t ix = std::min(static_cast<size_t>(std::round(src_x)), in_w - 1);
                            size_t in_idx = (b * channels + c) * in_h * in_w + iy * in_w + ix;
                            size_t out_idx = (b * channels + c) * out_h * out_w + oh * out_w + ow;
                            y[out_idx] = x[in_idx];
                        }
                    }
                }
            }
        }
    } else if (info.mode == InterpolateMode::LINEAR || info.mode == InterpolateMode::BILINEAR) {
        if (info.ndim == 1) {
            // Linear interpolation for 1D
            size_t in_w = info.input_shape[2];
            size_t out_w = info.output_shape[2];
            double scale = info.align_corners ? (in_w - 1.0) / (out_w - 1.0) : static_cast<double>(in_w) / out_w;

            for (size_t b = 0; b < batch; ++b) {
                for (size_t c = 0; c < channels; ++c) {
                    for (size_t ow = 0; ow < out_w; ++ow) {
                        double src_x = info.align_corners ? ow * scale : (ow + 0.5) * scale - 0.5;
                        src_x = std::max(0.0, std::min(src_x, static_cast<double>(in_w - 1)));
                        size_t x0 = static_cast<size_t>(std::floor(src_x));
                        size_t x1 = std::min(x0 + 1, in_w - 1);
                        double dx = src_x - x0;
                        T v0 = x[(b * channels + c) * in_w + x0];
                        T v1 = x[(b * channels + c) * in_w + x1];
                        y[(b * channels + c) * out_w + ow] = utils::cast<T>((1 - dx) * utils::cast<double>(v0) + dx * utils::cast<double>(v1));
                    }
                }
            }
        } else if (info.ndim == 2) {
            interpolate_bilinear_2d(x, y, batch, channels,
                                   info.input_shape[2], info.input_shape[3],
                                   info.output_shape[2], info.output_shape[3],
                                   info.align_corners);
        }
    } else if (info.mode == InterpolateMode::AREA) {
        // Area interpolation: average pooling
        size_t in_h = info.input_shape[2];
        size_t in_w = info.input_shape[3];
        size_t out_h = info.output_shape[2];
        size_t out_w = info.output_shape[3];
        double scale_h = static_cast<double>(in_h) / out_h;
        double scale_w = static_cast<double>(in_w) / out_w;

        for (size_t b = 0; b < batch; ++b) {
            for (size_t c = 0; c < channels; ++c) {
                for (size_t oh = 0; oh < out_h; ++oh) {
                    for (size_t ow = 0; ow < out_w; ++ow) {
                        double start_h = oh * scale_h;
                        double end_h = (oh + 1) * scale_h;
                        double start_w = ow * scale_w;
                        double end_w = (ow + 1) * scale_w;

                        size_t h0 = static_cast<size_t>(std::floor(start_h));
                        size_t h1 = static_cast<size_t>(std::ceil(end_h));
                        size_t w0 = static_cast<size_t>(std::floor(start_w));
                        size_t w1 = static_cast<size_t>(std::ceil(end_w));

                        T sum = utils::cast<T>(0.0);
                        size_t count = 0;
                        for (size_t ih = h0; ih < h1 && ih < in_h; ++ih) {
                            for (size_t iw = w0; iw < w1 && iw < in_w; ++iw) {
                                sum += x[(b * channels + c) * in_h * in_w + ih * in_w + iw];
                                count++;
                            }
                        }
                        y[(b * channels + c) * out_h * out_w + oh * out_w + ow] = count > 0 ? sum / utils::cast<T>(static_cast<double>(count)) : utils::cast<T>(0.0);
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
        interpolate_impl<fp16_t>(_info, reinterpret_cast<fp16_t *>(y), reinterpret_cast<const fp16_t *>(x));
        break;
    case INFINI_DTYPE_BF16:
        interpolate_impl<bf16_t>(_info, reinterpret_cast<bf16_t *>(y), reinterpret_cast<const bf16_t *>(x));
        break;
    case INFINI_DTYPE_F32:
        interpolate_impl<float>(_info, reinterpret_cast<float *>(y), reinterpret_cast<const float *>(x));
        break;
    case INFINI_DTYPE_F64:
        interpolate_impl<double>(_info, reinterpret_cast<double *>(y), reinterpret_cast<const double *>(x));
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::interpolate::cpu
