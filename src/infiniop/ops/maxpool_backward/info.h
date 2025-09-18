#ifndef __MAXPOOL_BACKWARD_INFO_H__
#define __MAXPOOL_BACKWARD_INFO_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"
#include <vector>

namespace op::maxpool_backward {

class MaxPoolBackwardInfo {
    MaxPoolBackwardInfo() = default;

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

    static utils::Result<MaxPoolBackwardInfo> create(
        infiniopTensorDescriptor_t grad_input_desc,  // gradient w.r.t. input
        infiniopTensorDescriptor_t grad_output_desc, // gradient w.r.t. output
        infiniopTensorDescriptor_t input_desc,       // original input
        void *kernel_size,
        void *strides,
        void *pads,
        bool ceil_mode) {

        MaxPoolBackwardInfo info;

        // Validate tensor dimensions
        if (input_desc->ndim() < 3 || input_desc->ndim() > 5) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        if (input_desc->ndim() != grad_input_desc->ndim() || grad_output_desc->ndim() != grad_input_desc->ndim()) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        // Check batch and channel dimensions match
        if (input_desc->dim(0) != grad_input_desc->dim(0) || input_desc->dim(1) != grad_input_desc->dim(1) || grad_output_desc->dim(0) != grad_input_desc->dim(0) || grad_output_desc->dim(1) != grad_input_desc->dim(1)) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        // Check spatial dimensions consistency
        for (size_t i = 2; i < input_desc->ndim(); ++i) {
            if (input_desc->dim(i) != grad_input_desc->dim(i)) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
        }

        info.ndim = input_desc->ndim() - 2; // spatial dimensions
        info.batch = input_desc->dim(0);
        info.channels = input_desc->dim(1);
        info.ceil_mode = ceil_mode;

        auto kernel_ptr = reinterpret_cast<const size_t *>(kernel_size);
        auto stride_ptr = reinterpret_cast<const size_t *>(strides);
        auto pad_ptr = reinterpret_cast<const size_t *>(pads);

        // Store spatial dimensions and pooling parameters
        for (size_t i = 0; i < info.ndim; ++i) {
            info.input_dims.push_back(input_desc->dim(i + 2));
            info.output_dims.push_back(grad_output_desc->dim(i + 2));
            info.kernel_sizes.push_back(kernel_ptr[i]);
            info.strides.push_back(stride_ptr[i]);
            info.pads.push_back(pad_ptr[i]);
        }

        return utils::Result<MaxPoolBackwardInfo>(std::move(info));
    }
};

} // namespace op::maxpool_backward

#endif // __MAXPOOL_BACKWARD_INFO_H__
