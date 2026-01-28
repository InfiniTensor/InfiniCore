#ifndef __MAX_POOL_INFO_H__
#define __MAX_POOL_INFO_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"
#include <vector>

namespace op::maxpool {

inline utils::Result<size_t> calculateMaxPoolOutputSize(
    size_t input_size,
    size_t kernel_size,
    size_t stride,
    size_t padding = 0,
    bool ceil_mode = false) {

    if (stride == 0) {
        return utils::Result<size_t>(INFINI_STATUS_BAD_PARAM);
    }
    if (kernel_size == 0) {
        return utils::Result<size_t>(INFINI_STATUS_BAD_PARAM);
    }

    // 理论最大输出数
    size_t max_output = 0;
    if (ceil_mode) {
        max_output = (input_size + 2 * padding - kernel_size + stride - 1) / stride + 1;
    } else {
        max_output = (input_size + 2 * padding - kernel_size) / stride + 1;
    }

    size_t valid_output = 0;
    for (size_t i = 0; i < max_output; ++i) {
        int64_t start = static_cast<int64_t>(i) * stride - padding;
        int64_t end = start + kernel_size;
        // 判断区间 [start, end) 和 [0, input_size) 是否有交集
        int64_t real_start = std::max(start, int64_t(0));
        int64_t real_end = std::min(end, int64_t(input_size));
        if (real_end > real_start) {
            ++valid_output;
        }
    }
    return utils::Result<size_t>(valid_output);
}

class MaxPoolInfo {
    MaxPoolInfo() = default;

public:
    std::vector<size_t> input_dims;
    std::vector<size_t> output_dims;
    std::vector<size_t> kernel_sizes;
    std::vector<size_t> strides;
    std::vector<size_t> pads;
    bool ceil_mode;
    size_t ndim;
    size_t batch;
    size_t channels;

    static utils::Result<MaxPoolInfo> create(
        infiniopTensorDescriptor_t output_desc,
        infiniopTensorDescriptor_t input_desc,
        void *kernel_size,
        void *strides,
        void *pads,
        bool ceil_mode) {

        MaxPoolInfo info;

        if (input_desc->ndim() < 3 || input_desc->ndim() > 5) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        if (input_desc->ndim() != output_desc->ndim()) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        if (input_desc->dim(0) != output_desc->dim(0) || input_desc->dim(1) != output_desc->dim(1)) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        info.ndim = input_desc->ndim() - 2; // spatial dimensions
        info.batch = input_desc->dim(0);
        info.channels = input_desc->dim(1);
        info.ceil_mode = ceil_mode;

        auto kernel_ptr = reinterpret_cast<const size_t *>(kernel_size);
        auto stride_ptr = reinterpret_cast<const size_t *>(strides);
        auto pad_ptr = reinterpret_cast<const size_t *>(pads);

        // Get spatial dimensions
        for (size_t i = 0; i < info.ndim; ++i) {
            info.input_dims.push_back(input_desc->dim(i + 2));
            info.kernel_sizes.push_back(kernel_ptr[i]);
            info.strides.push_back(stride_ptr[i]);
            info.pads.push_back(pad_ptr[i]);
            auto output_size = calculateMaxPoolOutputSize(
                info.input_dims[i], info.kernel_sizes[i], info.strides[i], info.pads[i], info.ceil_mode);
            CHECK_RESULT(output_size);
            size_t expected_size = output_size.take();
            if (expected_size != output_desc->dim(i + 2)) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }

            info.output_dims.push_back(output_desc->dim(i + 2));
        }
        return utils::Result<MaxPoolInfo>(std::move(info));
    }
};
} // namespace op::maxpool

#endif // __MAX_POOL_INFO_H__
