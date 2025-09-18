#ifndef __AVERAGEPOOL_BACKWARD_INFO_H__
#define __AVERAGEPOOL_BACKWARD_INFO_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"
#include <vector>

namespace op::averagepool_backward {

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

class AvgPoolBackwardInfo {
    AvgPoolBackwardInfo() = default;

public:
    std::vector<size_t> input_dims;  // original input dimensions
    std::vector<size_t> output_dims; // pooled output dimensions
    std::vector<size_t> kernel_sizes;
    std::vector<size_t> strides;
    std::vector<size_t> pads;
    bool ceil_mode;
    size_t ndim;
    size_t batch;
    size_t channels;
    bool has_implicit_padding = false;

    static utils::Result<AvgPoolBackwardInfo> create(
        infiniopTensorDescriptor_t grad_input_desc,  // gradient w.r.t. input
        infiniopTensorDescriptor_t grad_output_desc, // gradient w.r.t. output
        infiniopTensorDescriptor_t input_desc,       // original input from forward pass
        void *kernel_size,
        void *strides,
        void *pads,
        bool ceil_mode) {

        AvgPoolBackwardInfo info;

        if (input_desc->ndim() < 3 || input_desc->ndim() > 5) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        if (input_desc->ndim() != grad_input_desc->ndim() || grad_output_desc->ndim() != grad_input_desc->ndim()) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        if (input_desc->dim(0) != grad_input_desc->dim(0) || input_desc->dim(1) != grad_input_desc->dim(1) || grad_output_desc->dim(0) != grad_input_desc->dim(0) || grad_output_desc->dim(1) != grad_input_desc->dim(1)) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        for (size_t i = 2; i < input_desc->ndim(); ++i) {
            if (input_desc->dim(i) != grad_input_desc->dim(i)) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
        }

        info.ndim = input_desc->ndim() - 2;
        info.batch = input_desc->dim(0);
        info.channels = input_desc->dim(1);
        info.ceil_mode = ceil_mode;

        auto kernel_ptr = reinterpret_cast<const size_t *>(kernel_size);
        auto stride_ptr = reinterpret_cast<const size_t *>(strides);
        auto pad_ptr = reinterpret_cast<const size_t *>(pads);

        // 初始化隐式填充标志
        info.has_implicit_padding = false;
        for (size_t i = 0; i < info.ndim; ++i) {
            info.input_dims.push_back(input_desc->dim(i + 2));
            info.output_dims.push_back(grad_output_desc->dim(i + 2));
            info.kernel_sizes.push_back(kernel_ptr[i]);
            info.strides.push_back(stride_ptr[i]);
            info.pads.push_back(pad_ptr[i]);

            // 检查当前维度是否存在隐式填充
            if (hasImplicitPadding(info.input_dims[i], info.kernel_sizes[i],
                                   info.strides[i], info.pads[i], info.ceil_mode)) {
                info.has_implicit_padding = true;
            }
        }

        return utils::Result<AvgPoolBackwardInfo>(std::move(info));
    }
};

} // namespace op::averagepool_backward

#endif // __AVERAGEPOOL_BACKWARD_INFO_H__
