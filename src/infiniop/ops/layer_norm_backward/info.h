#ifndef __LAYER_NORM_BACKWARD_INFO_H__
#define __LAYER_NORM_BACKWARD_INFO_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"

namespace op::layer_norm_backward {

class LayerNormBackwardInfo {
private:
    LayerNormBackwardInfo() = default;

public:
//  ---------------------------- start: define member variables of Info ----------------------------
    infiniDtype_t dtype;
	size_t batch_size;
	size_t channel_size;
	size_t feature_size;

    std::vector<ptrdiff_t> grad_input_strides;
    std::vector<ptrdiff_t> grad_weight_strides;
    std::vector<ptrdiff_t> grad_bias_strides;
    std::vector<ptrdiff_t> grad_output_strides;
    std::vector<ptrdiff_t> weight_strides;
    std::vector<ptrdiff_t> input_standardization_strides;
    std::vector<ptrdiff_t> input_std_deviation_strides;
    bool bias;

//  ----------------------------- end: define member variables of Info -----------------------------

    static utils::Result<LayerNormBackwardInfo> createLayerNormBackwardInfo(
        infiniopTensorDescriptor_t grad_input_desc,
        infiniopTensorDescriptor_t grad_weight_desc,
        infiniopTensorDescriptor_t grad_bias_desc,
        infiniopTensorDescriptor_t grad_output_desc,
        infiniopTensorDescriptor_t weight_desc,
        infiniopTensorDescriptor_t input_standardization_desc,
        infiniopTensorDescriptor_t input_std_deviation_desc
    ) {
//  ------------------------- start: check tensor shape and input validity -------------------------
        CHECK_SAME_SHAPE(
            grad_input_desc->shape(), input_standardization_desc->shape(), grad_output_desc->shape()
        );
        CHECK_SAME_SHAPE(grad_weight_desc->shape(), weight_desc->shape());
        size_t batch_size = grad_input_desc->dim(0),
            channel_size = grad_input_desc->dim(1),
            feature_size = grad_input_desc->dim(2);
        CHECK_OR_RETURN(
            (weight_desc->ndim() == 1) && (weight_desc->dim(0) == feature_size),
            INFINI_STATUS_BAD_TENSOR_SHAPE
        );
        bool bias = grad_bias_desc != nullptr;
        CHECK_OR_RETURN(
            (!bias) || (grad_bias_desc->ndim() == 1 && grad_bias_desc->dim(0) == feature_size),
            INFINI_STATUS_BAD_TENSOR_SHAPE
        );
        CHECK_OR_RETURN(
            input_std_deviation_desc->ndim() == 2 && \
            input_std_deviation_desc->dim(0) == batch_size && \
            input_std_deviation_desc->dim(1) == channel_size,
            INFINI_STATUS_BAD_TENSOR_SHAPE
        );
        
//  -------------------------- end: check tensor shape and input validity --------------------------
        return utils::Result<LayerNormBackwardInfo>(LayerNormBackwardInfo{
//  ------------------------------ start: create an instance of Info -------------------------------
            grad_input_desc->dtype(),
            batch_size, channel_size, feature_size,
            grad_input_desc->strides(),
            grad_weight_desc->strides(),
            bias ? grad_bias_desc->strides() : std::vector<ptrdiff_t>(),
            grad_output_desc->strides(),
            weight_desc->strides(),
            input_standardization_desc->strides(),
            input_std_deviation_desc->strides(),
            bias
//  ------------------------------- end: create an instance of Info --------------------------------
        });
    }
};
}

#endif //  __LAYER_NORM_BACKWARD_INFO_H__
