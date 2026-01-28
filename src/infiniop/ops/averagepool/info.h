#ifndef __AVERAGEPOOL_INFO_H__
#define __AVERAGEPOOL_INFO_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"
#include <cstddef>
#include <vector>

namespace op::averagepool {

inline utils::Result<size_t> calculatePoolOutputSize(
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

    size_t padded_input_size = input_size + 2 * padding;

    if (padded_input_size < kernel_size) {
        return utils::Result<size_t>(INFINI_STATUS_BAD_TENSOR_SHAPE);
    }

    size_t output_size;
    if (ceil_mode) {
        // 等效于整数的上取整
        output_size = (padded_input_size - kernel_size + stride - 1) / stride + 1;
    } else {
        // 等效于整数的下取整
        output_size = (padded_input_size - kernel_size) / stride + 1;
    }

    return utils::Result<size_t>(output_size);
}

// 检查是否存在隐式填充
inline bool hasImplicitPadding(
    size_t input_size,
    size_t kernel_size,
    size_t stride,
    size_t padding,
    bool ceil_mode) {

    if (!ceil_mode) {
        return false;
    }
    return ((input_size + 2 * padding) - kernel_size) % stride != 0;
}

class AvgPoolInfo {
    AvgPoolInfo() = default;

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
    bool has_implicit_padding = false;

    static utils::Result<AvgPoolInfo> create(
        infiniopTensorDescriptor_t output_desc,
        infiniopTensorDescriptor_t input_desc,
        void *kernel_size,
        void *strides,
        void *pads,
        bool ceil_mode) {

        AvgPoolInfo info;

        if (input_desc->ndim() < 3 || input_desc->ndim() > 5) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        if (input_desc->ndim() != output_desc->ndim()) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        if (input_desc->dim(0) != output_desc->dim(0) || input_desc->dim(1) != output_desc->dim(1)) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        info.ndim = input_desc->ndim() - 2; // 空间维度
        info.batch = input_desc->dim(0);
        info.channels = input_desc->dim(1);
        info.ceil_mode = ceil_mode;

        auto kernel_ptr = reinterpret_cast<const size_t *>(kernel_size);
        auto stride_ptr = reinterpret_cast<const size_t *>(strides);
        auto pad_ptr = reinterpret_cast<const size_t *>(pads);

        // 初始化隐式填充标志
        info.has_implicit_padding = false;

        // 获取并校验空间维度
        for (size_t i = 0; i < info.ndim; ++i) {
            info.input_dims.push_back(input_desc->dim(i + 2));
            info.kernel_sizes.push_back(kernel_ptr[i]);
            info.strides.push_back(stride_ptr[i]);
            info.pads.push_back(pad_ptr[i]);

            auto output_size_result = calculatePoolOutputSize(
                info.input_dims[i], info.kernel_sizes[i], info.strides[i], info.pads[i], info.ceil_mode);
            CHECK_RESULT(output_size_result);

            size_t expected_size = output_size_result.take();
            if (expected_size != output_desc->dim(i + 2)) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }

            info.output_dims.push_back(output_desc->dim(i + 2));

            // 检查当前维度是否存在隐式填充
            if (hasImplicitPadding(info.input_dims[i], info.kernel_sizes[i],
                                   info.strides[i], info.pads[i], info.ceil_mode)) {
                info.has_implicit_padding = true;
            }
        }
        return utils::Result<AvgPoolInfo>(std::move(info));
    }
};
} // namespace op::averagepool

#endif // __AVERAGEPOOL_INFO_H__
