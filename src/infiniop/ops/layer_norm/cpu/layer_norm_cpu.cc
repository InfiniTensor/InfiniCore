#include "layer_norm_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../../reduce/cpu/reduce.h"
#include "../info.h"

namespace op::layer_norm::cpu {

template <typename Tdata>
infiniStatus_t calculate_layer_norm(
    const LayerNormInfo &info,
	Tdata * output,
	Tdata * input_standardization,
	Tdata * input_std_deviation,
	const Tdata * input,
	const Tdata * weight,
	const Tdata * bias
) {
//  -------------------------------- start: perform operator on CPU --------------------------------
    #pragma omp parallel for
    for(size_t b = 0; b < info.input_shape[0] * info.input_shape[1]; b ++)
    {
        size_t b0 = b / info.input_shape[1], b1 = b % info.input_shape[1];
        auto output_ptr = output + b0 * info.output_strides[0] + b1 * info.output_strides[1];
        auto input_ptr = input + b0 * info.input_strides[0] + b1 * info.input_strides[1];
        auto standard_ptr = input_standardization + b0 * info.input_standardization_strides[0] + b1 * info.input_standardization_strides[1];
        auto std_ptr = input_std_deviation + b0 * info.input_std_deviation_strides[0] + b1 * info.input_std_deviation_strides[1];
        float mean = op::common_cpu::reduce_op::sum(
            input_ptr,
            info.normalized_size,
            info.input_strides[2]
        ) / info.input_shape[2];
        float sum_sq = op::common_cpu::reduce_op::sumSquared(
            input_ptr,
            info.normalized_size,
            info.input_strides[2]
        );
        float var = sum_sq / (info.normalized_size) - mean * mean;
        float std_deviation = std::sqrt(var + info.eps);
        *std_ptr = utils::cast<Tdata>(std_deviation);

        for(size_t d = 0; d < info.normalized_size; d ++)
        {
            float x_standard = (utils::cast<float>(*(input_ptr + d * info.input_strides[2])) - mean) / std_deviation;
            *(standard_ptr + d * info.input_standardization_strides[2]) = utils::cast<Tdata>(x_standard);
            *(output_ptr + d * info.output_strides[2]) = utils::cast<Tdata>(
                x_standard * utils::cast<float>(*(weight + d * info.weight_strides[0])) + \
                (info.bias_exist ? utils::cast<float>(*(bias + d * info.bias_strides[0])) : float(0))
            );
        }           

        
    
}
//  --------------------------------- end: perform operator on CPU ---------------------------------
    return INFINI_STATUS_SUCCESS;
}

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
	infiniopTensorDescriptor_t output_desc,
	infiniopTensorDescriptor_t input_standardization_desc,
	infiniopTensorDescriptor_t input_std_deviation_desc,
	infiniopTensorDescriptor_t input_desc,
	infiniopTensorDescriptor_t weight_desc,
	infiniopTensorDescriptor_t bias_desc,
	float eps
) {
    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);

//  --------------------- start: check data type and calculate workspace size ----------------------
    auto dtype = input_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);
    size_t WorkSpaceSize = 0;
//  ---------------------- end: check data type and calculate workspace size -----------------------

    auto result = LayerNormInfo::createLayerNormInfo(
		output_desc,
		input_standardization_desc,
		input_std_deviation_desc,
		input_desc,
		weight_desc,
		bias_desc,
		eps
    );
    CHECK_RESULT(result);
    const LayerNormInfo &info = result.take();
    
    *desc_ptr = new Descriptor(
        dtype, std::move(info), WorkSpaceSize,
        nullptr,
        handle->device, handle->device_id
    );    

    return INFINI_STATUS_SUCCESS;
}

#define CALCULATE_LAYER_NORM(TDATA) \
    CHECK_STATUS(calculate_layer_norm<TDATA>(_info, \
(TDATA *)output, (TDATA *)input_standardization, (TDATA *)input_std_deviation, (const TDATA *)input, (const TDATA *)weight, (const TDATA *)bias))

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
	void * output,
	void * input_standardization,
	void * input_std_deviation,
	const void * input,
	const void * weight,
	const void * bias,
    void *stream
) const {
    if (_info.dtype == INFINI_DTYPE_F16) {
        CALCULATE_LAYER_NORM(fp16_t);
    } else if (_info.dtype == INFINI_DTYPE_BF16) {
        CALCULATE_LAYER_NORM(bf16_t);
    } else if (_info.dtype == INFINI_DTYPE_F32) {
        CALCULATE_LAYER_NORM(float);
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    return INFINI_STATUS_SUCCESS;
}
}
