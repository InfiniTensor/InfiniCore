#ifndef __CONV_BACKWARD_INFO_H__
#define __CONV_BACKWARD_INFO_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"
#include <vector>

namespace op::conv_backward {

class ConvBackwardInfo {
    ConvBackwardInfo() = default;

public:
    size_t ndim;
    size_t batch;
    size_t in_channels;
    size_t out_channels;
    size_t groups;
    std::vector<size_t> input_dims;
    std::vector<size_t> weight_dims;
    std::vector<size_t> grad_output_dims;
    std::vector<size_t> pads;
    std::vector<size_t> strides;
    std::vector<size_t> dilations;

    static utils::Result<ConvBackwardInfo> create(
        infiniopTensorDescriptor_t grad_output_desc,
        infiniopTensorDescriptor_t input_desc,
        infiniopTensorDescriptor_t weight_desc,
        void *pads,
        void *strides,
        void *dilations,
        size_t groups) {
        ConvBackwardInfo info;
        info.ndim = input_desc->ndim() - 2;
        info.batch = input_desc->dim(0);
        info.in_channels = input_desc->dim(1);
        info.out_channels = weight_desc->dim(0);
        info.groups = groups;
        // 校验维度
        if (input_desc->ndim() != weight_desc->ndim() || input_desc->ndim() != grad_output_desc->ndim()) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (input_desc->dim(0) != grad_output_desc->dim(0) || weight_desc->dim(0) != grad_output_desc->dim(1)) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        for (size_t i = 2; i < input_desc->ndim(); ++i) {
            info.input_dims.push_back(input_desc->dim(i));
            info.weight_dims.push_back(weight_desc->dim(i));
            info.grad_output_dims.push_back(grad_output_desc->dim(i));
        }

        auto pad_ptr = reinterpret_cast<const int *>(pads);
        auto stride_ptr = reinterpret_cast<const int *>(strides);
        auto dilation_ptr = reinterpret_cast<const int *>(dilations);

        for (size_t i = 0; i < info.ndim; ++i) {
            info.pads.push_back(pad_ptr ? static_cast<size_t>(pad_ptr[i]) : 0);
            info.strides.push_back(stride_ptr ? static_cast<size_t>(stride_ptr[i]) : 1);
            info.dilations.push_back(dilation_ptr ? static_cast<size_t>(dilation_ptr[i]) : 1);
        }
        return utils::Result<ConvBackwardInfo>(std::move(info));
    }
};

} // namespace op::conv_backward

#endif // __CONV_BACKWARD_INFO_H__
