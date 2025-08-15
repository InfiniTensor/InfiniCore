#ifndef __LAYER_NORM_INFO_H__
#define __LAYER_NORM_INFO_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"

namespace op::layer_norm {

class LayerNormInfo {
private:
    LayerNormInfo() = default;

public:
//  ---------------------------- start: define member variables of Info ----------------------------
    infiniDtype_t dtype;
	size_t ndim;
	std::vector<size_t> input_shape;
	size_t normalized_size;
	std::vector<ptrdiff_t> output_strides;
	std::vector<ptrdiff_t> input_standardization_strides;
	std::vector<ptrdiff_t> input_std_deviation_strides;
	std::vector<ptrdiff_t> input_strides;
	std::vector<ptrdiff_t> weight_strides;
	std::vector<ptrdiff_t> bias_strides;
	float eps;
	bool bias_exist;

//  ----------------------------- end: define member variables of Info -----------------------------

    static utils::Result<LayerNormInfo> createLayerNormInfo(
		infiniopTensorDescriptor_t output_desc,
		infiniopTensorDescriptor_t input_standardization_desc,
		infiniopTensorDescriptor_t input_std_deviation_desc,
		infiniopTensorDescriptor_t input_desc,
		infiniopTensorDescriptor_t weight_desc,
		infiniopTensorDescriptor_t bias_desc,
		float eps
    ) {
//  ------------------------- start: check tensor shape and input validity -------------------------
		CHECK_SAME_SHAPE(
            output_desc->shape(), input_desc->shape(), input_standardization_desc->shape()
        );
        size_t batch_size = input_desc->dim(0),
            channel_size = input_desc->dim(1),
            feature_size = input_desc->dim(2);	
			
		bool bias_exist = bias_desc != nullptr;
		CHECK_OR_RETURN(
            (!bias_exist) || (bias_desc->ndim() == 1 && bias_desc->dim(0) == feature_size),
            INFINI_STATUS_BAD_TENSOR_SHAPE
        );
		CHECK_OR_RETURN(
            (weight_desc->ndim() == 1) && (weight_desc->dim(0) == feature_size),
            INFINI_STATUS_BAD_TENSOR_SHAPE
        );
        CHECK_OR_RETURN(
            input_std_deviation_desc->ndim() == 2 && \
			input_std_deviation_desc->dim(0) == batch_size && \
			input_std_deviation_desc->dim(1) == channel_size,
            INFINI_STATUS_BAD_TENSOR_SHAPE
        );
//  -------------------------- end: check tensor shape and input validity --------------------------
        return utils::Result<LayerNormInfo>(LayerNormInfo{
//  ------------------------------ start: create an instance of Info -------------------------------
            output_desc->dtype(),
			input_desc->ndim(),
			input_desc->shape(),
			input_desc->dim(input_desc->ndim() - 1),
			output_desc->strides(),
			input_standardization_desc->strides(),
			input_std_deviation_desc->strides(),
			input_desc->strides(),
			weight_desc->strides(),
			bias_exist ? bias_desc->strides() : std::vector<ptrdiff_t>(),
			eps,
			bias_exist
//  ------------------------------- end: create an instance of Info --------------------------------
        });
    }
};
}

#endif //  __LAYER_NORM_INFO_H__
