#include "pad_cpu.h"
#include "../../../utils.h"
#include <cstring>
#include <algorithm>
#include <cmath>

namespace op::pad::cpu {

PadMode parseMode(const char *mode_str) {
    if (std::strcmp(mode_str, "constant") == 0) {
        return PadMode::CONSTANT;
    } else if (std::strcmp(mode_str, "reflect") == 0) {
        return PadMode::REFLECT;
    } else if (std::strcmp(mode_str, "replicate") == 0) {
        return PadMode::REPLICATE;
    } else if (std::strcmp(mode_str, "circular") == 0) {
        return PadMode::CIRCULAR;
    }
    return PadMode::CONSTANT;  // Default
}

utils::Result<PadInfo> PadInfo::create(
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t y_desc,
    const void *pad,
    size_t pad_size,
    const char *mode_str,
    double value) {

    auto x_shape = x_desc->shape();
    auto y_shape = y_desc->shape();
    size_t ndim = x_desc->ndim();

    // Parse pad array
    const int *pad_array = reinterpret_cast<const int *>(pad);
    size_t pad_len = pad_size / sizeof(int);

    // Pad array should have 2*ndim elements (left and right for each dimension)
    // But it might be shorter (only last dimensions)
    std::vector<int> pads(2 * ndim, 0);
    if (pad_len == 2 * ndim) {
        // Full pad specification
        std::memcpy(pads.data(), pad_array, pad_len * sizeof(int));
    } else if (pad_len == 2) {
        // Only last dimension
        pads[2 * (ndim - 1)] = pad_array[0];
        pads[2 * (ndim - 1) + 1] = pad_array[1];
    } else if (pad_len % 2 == 0 && pad_len <= 2 * ndim) {
        // Last few dimensions
        size_t start_dim = ndim - pad_len / 2;
        for (size_t i = 0; i < pad_len; ++i) {
            pads[2 * start_dim + i] = pad_array[i];
        }
    } else {
        return INFINI_STATUS_BAD_PARAM;
    }

    // Calculate expected output shape
    std::vector<size_t> expected_output_shape = x_shape;
    for (size_t i = 0; i < ndim; ++i) {
        expected_output_shape[i] += pads[2 * i] + pads[2 * i + 1];
    }

    if (y_shape != expected_output_shape) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    PadInfo info;
    info.ndim = ndim;
    info.input_shape = x_shape;
    info.output_shape = y_shape;
    info.pads = pads;
    info.mode = parseMode(mode_str);
    info.value = value;

    return utils::Result<PadInfo>(std::move(info));
}

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    const void *pad,
    size_t pad_size,
    const char *mode,
    double value) {

    auto dtype = x_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64, INFINI_DTYPE_BF16);

    auto info_result = PadInfo::create(x_desc, y_desc, pad, pad_size, mode, value);
    CHECK_RESULT(info_result);

    *desc_ptr = new Descriptor(dtype, info_result.take(), handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <typename T>
void pad_impl(
    const PadInfo &info,
    T *y,
    const T *x) {

    size_t output_size = 1;
    for (size_t i = 0; i < info.ndim; ++i) {
        output_size *= info.output_shape[i];
    }

    // Initialize output with padding value (for constant mode)
    if (info.mode == PadMode::CONSTANT) {
        T pad_value = utils::cast<T>(info.value);
        std::fill(y, y + output_size, pad_value);
    }

    // Helper function to map output index to input index
    auto getInputIndex = [&](const std::vector<size_t> &out_coords) -> std::pair<bool, size_t> {
        std::vector<size_t> in_coords(info.ndim);
        bool valid = true;

        for (size_t d = 0; d < info.ndim; ++d) {
            int pad_left = info.pads[2 * d];
            int pad_right = info.pads[2 * d + 1];
            size_t out_idx = out_coords[d];
            size_t in_size = info.input_shape[d];

            if (out_idx < static_cast<size_t>(pad_left)) {
                // Left padding
                if (info.mode == PadMode::CONSTANT) {
                    valid = false;
                    break;
                } else if (info.mode == PadMode::REFLECT) {
                    in_coords[d] = pad_left - out_idx;
                } else if (info.mode == PadMode::REPLICATE) {
                    in_coords[d] = 0;
                } else if (info.mode == PadMode::CIRCULAR) {
                    in_coords[d] = in_size - (pad_left - out_idx);
                }
            } else if (out_idx >= pad_left + in_size) {
                // Right padding
                if (info.mode == PadMode::CONSTANT) {
                    valid = false;
                    break;
                } else {
                    size_t excess = out_idx - (pad_left + in_size);
                    if (info.mode == PadMode::REFLECT) {
                        in_coords[d] = in_size - 2 - excess;
                    } else if (info.mode == PadMode::REPLICATE) {
                        in_coords[d] = in_size - 1;
                    } else if (info.mode == PadMode::CIRCULAR) {
                        in_coords[d] = excess;
                    }
                }
            } else {
                // Inside input range
                in_coords[d] = out_idx - pad_left;
            }

            // Bounds checking for reflect mode
            if (info.mode == PadMode::REFLECT) {
                while (in_coords[d] >= in_size) {
                    in_coords[d] = 2 * (in_size - 1) - in_coords[d];
                }
            }
        }

        if (!valid) {
            return {false, 0};
        }

        // Convert coordinates to linear index
        size_t in_index = 0;
        size_t stride = 1;
        for (size_t d = info.ndim; d-- > 0;) {
            in_index += in_coords[d] * stride;
            stride *= info.input_shape[d];
        }

        return {true, in_index};
    };

    // Iterate over output tensor
    std::vector<size_t> out_coords(info.ndim, 0);
    for (size_t out_idx = 0; out_idx < output_size; ++out_idx) {
        // Convert linear index to coordinates
        size_t temp = out_idx;
        for (size_t d = info.ndim; d-- > 0;) {
            out_coords[d] = temp % info.output_shape[d];
            temp /= info.output_shape[d];
        }

        auto [valid, in_idx] = getInputIndex(out_coords);
        if (valid) {
            y[out_idx] = x[in_idx];
        }
        // For constant mode, value is already set
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
        pad_impl<fp16_t>(_info, reinterpret_cast<fp16_t *>(y), reinterpret_cast<const fp16_t *>(x));
        break;
    case INFINI_DTYPE_BF16:
        pad_impl<bf16_t>(_info, reinterpret_cast<bf16_t *>(y), reinterpret_cast<const bf16_t *>(x));
        break;
    case INFINI_DTYPE_F32:
        pad_impl<float>(_info, reinterpret_cast<float *>(y), reinterpret_cast<const float *>(x));
        break;
    case INFINI_DTYPE_F64:
        pad_impl<double>(_info, reinterpret_cast<double *>(y), reinterpret_cast<const double *>(x));
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::pad::cpu
