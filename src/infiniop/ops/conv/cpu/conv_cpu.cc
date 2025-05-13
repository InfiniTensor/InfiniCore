#include "conv_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include <algorithm>

namespace op::conv::cpu {

inline size_t calculatePaddedInputSize(const ConvInfo &info) {
    std::vector<size_t> shape(info.ndim + 2);
    shape[0] = info.batch;
    shape[1] = info.in_channels;
    for (size_t i = 0; i < info.ndim; ++i) {
        shape[i + 2] = info.input_dims[i];
    }

    return op::common_cpu::getPaddedSize(info.ndim + 2, shape.data(), info.pads_info.data());
}

inline size_t calculateOutputSize(const ConvInfo &info) {
    size_t size = info.batch * info.out_channels;
    for (size_t i = 0; i < info.ndim; ++i) {
        size *= info.output_dims[i];
    }
    return size;
}

inline bool needsPadding(const ConvInfo &info) {
    return std::any_of(info.pads_info.begin(), info.pads_info.end(),
                       [](size_t pad) { return pad > 0; });
}

inline size_t getDtypeSize(infiniDtype_t dtype) {
    switch (dtype) {
    case INFINI_DTYPE_F16:
        return 2;
    case INFINI_DTYPE_F32:
        return 4;
    default:
        return 0;
    }
}

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t w_desc,
    const void *pads,
    const void *strides,
    const void *dilations,
    size_t n) {
    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto dtype = y_desc->dtype();

    if (dtype != INFINI_DTYPE_F16 && dtype != INFINI_DTYPE_F32) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    auto result = ConvInfo::create(handle_, y_desc, x_desc, w_desc,
                                   pads, strides, dilations, n);
    CHECK_RESULT(result);

    size_t WorkSpaceSize = 0;
    ConvInfo &info = *result;

    if (needsPadding(info)) {
        WorkSpaceSize += calculatePaddedInputSize(info) * getDtypeSize(dtype);
    }

    if (dtype == INFINI_DTYPE_F16) {
        WorkSpaceSize += calculateOutputSize(info) * sizeof(float);
    }

    *desc_ptr = new Descriptor(
        dtype, result.take(), WorkSpaceSize,
        nullptr,
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <typename Tdata>
void fillPaddedInput(
    const ConvInfo &info,
    const Tdata *x,
    const size_t *padded_x_shape,
    Tdata *padded_x,
    size_t x_index,
    size_t padded_x_index,
    size_t ndim) {
    size_t x_shape_val;
    if (ndim == 0) {
        x_shape_val = info.batch;
    } else if (ndim == 1) {
        x_shape_val = info.in_channels;
    } else {
        x_shape_val = info.input_dims[ndim - 2];
    }

    const auto padded_x_shape_val = padded_x_shape[ndim];
    const auto x_base_index = x_index * x_shape_val;

    size_t pad_offset = 0;
    if (ndim >= 2 && x_shape_val != padded_x_shape_val) {
        pad_offset = info.pads_info[ndim - 2];
    }

    const auto padded_x_base_index = padded_x_index * padded_x_shape_val + pad_offset;

    for (size_t i = 0; i < x_shape_val; ++i) {
        if (ndim == info.ndim + 2 - 1) {
            padded_x[padded_x_base_index + i] = x[x_base_index + i];
        } else {
            fillPaddedInput(info, x, padded_x_shape, padded_x,
                            x_base_index + i, padded_x_base_index + i, ndim + 1);
        }
    }
}

template <typename Xdata, typename Ydata>
void _applyConv(
    const ConvInfo &info,
    Ydata *y,
    const Xdata *x,
    const Xdata *w,
    const size_t *x_shape,
    size_t x_index,
    size_t w_index,
    size_t y_index,
    size_t ndim) {

    size_t dim_size, kernel_size;
    size_t dilation, stride;

    if (ndim < 2) {
        return;
    } else {
        dim_size = x_shape[ndim];
        kernel_size = info.kernel_dims[ndim - 2];
        dilation = info.dilations_info[ndim - 2];
        stride = info.strides_info[ndim - 2];
    }
    const auto steps = (dim_size - dilation * (kernel_size - 1) - 1) / stride + 1;
    x_index *= dim_size;
    w_index *= kernel_size;
    size_t y_stride;
    if (ndim == 0) {
        y_stride = info.out_channels;
    } else if (ndim == 1) {
        y_stride = 1;
    } else {
        y_stride = info.output_dims[ndim - 2];
    }
    y_index *= y_stride;
    for (size_t i = 0; i < steps; ++i, ++y_index) {
        for (size_t k = 0; k < kernel_size; ++k) {
            const auto curr_x_index = x_index + i * stride + k * dilation;
            const auto curr_w_index = w_index + k;
            if (ndim == info.ndim + 1) {
                if constexpr (std::is_same<Xdata, fp16_t>::value) {
                    y[y_index] += utils::cast<float>(x[curr_x_index]) * utils::cast<float>(w[curr_w_index]);
                } else {
                    y[y_index] += x[curr_x_index] * w[curr_w_index];
                }
            } else {
                _applyConv(info, y, x, w, x_shape, curr_x_index, curr_w_index,
                           y_index, ndim + 1);
            }
        }
    }
}

template <typename Xdata, typename Ydata>
void applyConv(
    const ConvInfo &info,
    Ydata *y,
    const Xdata *x,
    const Xdata *w,
    const size_t *x_shape) {
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(info.batch); ++i) {
        for (ptrdiff_t j = 0; j < static_cast<ptrdiff_t>(info.out_channels); ++j) {
            size_t y_index = i * info.out_channels + j;
            for (size_t k = 0; k < info.in_channels; ++k) {
                size_t x_index = i * info.in_channels + k;
                size_t w_index = j * info.in_channels + k;
                _applyConv(info, y, x, w, x_shape, x_index, w_index, y_index, 2);
            }
        }
    }
}

template <typename Xdata, typename Ydata>
void _conv_cpu(
    const ConvInfo &info,
    void *workspace,
    size_t workspace_size,
    Ydata *y,
    const Xdata *x,
    const Xdata *w) {
    if (needsPadding(info)) {
        auto padded_x = reinterpret_cast<Xdata *>(workspace);
        std::vector<size_t> padded_shape(info.ndim + 2);
        padded_shape[0] = info.batch;
        padded_shape[1] = info.in_channels;
        for (size_t i = 0; i < info.ndim; ++i) {
            padded_shape[i + 2] = info.input_dims[i] + 2 * info.pads_info[i];
        }
        if constexpr (std::is_same<Xdata, fp16_t>::value) {
            fp16_t zero_val = utils::cast<fp16_t>(0.0f);
            std::fill(padded_x, padded_x + calculatePaddedInputSize(info), zero_val);
        } else if constexpr (std::is_same<Xdata, float>::value) {
            std::fill(padded_x, padded_x + calculatePaddedInputSize(info), 0.0f);
        } else {
            std::fill(padded_x, padded_x + calculatePaddedInputSize(info), static_cast<Xdata>(0));
        }
        fillPaddedInput(info, x, padded_shape.data(), padded_x, 0, 0, 0);

        applyConv(info, y, padded_x, w, padded_shape.data());
    } else {
        std::vector<size_t> shape(info.ndim + 2);
        shape[0] = info.batch;
        shape[1] = info.in_channels;
        for (size_t i = 0; i < info.ndim; ++i) {
            shape[i + 2] = info.input_dims[i];
        }
        applyConv(info, y, x, w, shape.data());
    }
}

template <typename Tdata>
infiniStatus_t conv_cpu(
    const ConvInfo &info,
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    const void *w) {
    auto y_ptr = reinterpret_cast<Tdata *>(y);
    auto x_ptr = reinterpret_cast<const Tdata *>(x);
    auto w_ptr = reinterpret_cast<const Tdata *>(w);
    if constexpr (std::is_same<Tdata, float>::value) {
        std::fill(y_ptr, y_ptr + calculateOutputSize(info), 0.0f);
    } else if constexpr (std::is_same<Tdata, fp16_t>::value) {
        fp16_t zero_val = utils::cast<fp16_t>(0.0f);
        std::fill(y_ptr, y_ptr + calculateOutputSize(info), zero_val);
    } else {
        std::fill(y_ptr, y_ptr + calculateOutputSize(info), static_cast<Tdata>(0));
    }
    _conv_cpu<Tdata, Tdata>(info, workspace, workspace_size, y_ptr, x_ptr, w_ptr);

    return INFINI_STATUS_SUCCESS;
}

template <>
infiniStatus_t conv_cpu<fp16_t>(
    const ConvInfo &info,
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    const void *w) {
    auto y_float = reinterpret_cast<float *>(workspace);
    auto x_half = reinterpret_cast<const fp16_t *>(x);
    auto w_half = reinterpret_cast<const fp16_t *>(w);

    std::fill(y_float, y_float + calculateOutputSize(info), 0.0f);

    void *conv_workspace = y_float + calculateOutputSize(info);
    size_t conv_workspace_size = workspace_size - calculateOutputSize(info) * sizeof(float);

    _conv_cpu<fp16_t, float>(info, conv_workspace, conv_workspace_size, y_float, x_half, w_half);

    auto y_half = reinterpret_cast<fp16_t *>(y);
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(calculateOutputSize(info)); ++i) {
        y_half[i] = utils::cast<fp16_t>(y_float[i]);
    }

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    const void *w,
    void *stream) const {
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_BAD_PARAM;
    }
    switch (_dtype) {
    case INFINI_DTYPE_F16:
        return conv_cpu<fp16_t>(_info, workspace, workspace_size, y, x, w);
    case INFINI_DTYPE_F32:
        return conv_cpu<float>(_info, workspace, workspace_size, y, x, w);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::conv::cpu
