#include "batch_norm_backward_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../../reduce/cpu/reduce.h"
#include "../info.h"

namespace op::batch_norm_backward::cpu {

template <typename Tdata>
infiniStatus_t calculate_batch_norm_backward(
    const BatchNormBackwardInfo &info,
	Tdata * grad_input,
	Tdata * grad_weight,
	Tdata * grad_bias,
	const Tdata * input,
	const Tdata * grad_output,
	const Tdata * weight,
	const Tdata * running_mean,
	const Tdata * running_var
) {
//  -------------------------------- start: perform operator on CPU --------------------------------
    #pragma omp parallel for
    for(size_t c = 0; c < info.channel_size; c++)
    {
        double dbias = 0., dweight = 0., dvar = 0., dmu0 = 0., dmu1 = 0.;
        // double std = std::sqrt();
        double var = utils::cast<double>(*(running_var + c * info.running_var_stride));
        double std = std::sqrt(var);
        double mean = utils::cast<double>(*(running_mean + c * info.running_mean_stride));
        double wt = utils::cast<double>(*(weight + c * info.weight_stride));
        for (size_t b = 0; b < info.batch_size; b ++) {
            for(size_t d = 0; d < info.dim_size; d ++) {
                size_t index = b * (info.channel_size * info.dim_size) + c * info.dim_size + d;
                dbias += utils::cast<double>(grad_output[index]);
                dweight += utils::cast<double>(grad_output[index]) * (
                    utils::cast<double>(input[index]) - mean
                );
                double dx_norm = utils::cast<double>(grad_output[index]) * wt;
                double x_centered = utils::cast<double>(input[index]) - mean;
                dvar += dx_norm * x_centered * (-0.5) / (std * var);
                dmu0 += dx_norm * (-1) / std;
                dmu1 += (-2) * x_centered;
            }
        }
        *(grad_bias + c * info.grad_bias_stride) = utils::cast<Tdata>(dbias);
        *(grad_weight + c * info.grad_weight_stride) = utils::cast<Tdata>(dweight / std); 
        for(size_t b = 0; b < info.batch_size; b++)
        {
            for(size_t d = 0; d < info.dim_size; d++)
            {
                size_t index = b * (info.channel_size * info.dim_size) + c * info.dim_size + d;
                double dx_norm = utils::cast<double>(grad_output[index]) * wt;
                double x_centered = utils::cast<double>(input[index]) - mean;
                grad_input[index] = utils::cast<Tdata>(
                    dx_norm / std + \
                    ((dvar * 2 * x_centered) + (dmu0 + dmu1 * dvar)) / (info.batch_size * info.dim_size)
                );                
            }
        }
    }
//  --------------------------------- end: perform operator on CPU ---------------------------------
    return INFINI_STATUS_SUCCESS;
}

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
	infiniopTensorDescriptor_t grad_input_desc,
	infiniopTensorDescriptor_t grad_weight_desc,
	infiniopTensorDescriptor_t grad_bias_desc,
	infiniopTensorDescriptor_t input_desc,
	infiniopTensorDescriptor_t grad_output_desc,
	infiniopTensorDescriptor_t weight_desc,
	infiniopTensorDescriptor_t running_mean_desc,
	infiniopTensorDescriptor_t running_var_desc
) {
    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);

//  --------------------- start: check data type and calculate workspace size ----------------------
    auto dtype = grad_input_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);
    size_t WorkSpaceSize = 0;
//  ---------------------- end: check data type and calculate workspace size -----------------------

    auto result = BatchNormBackwardInfo::createBatchNormBackwardInfo(
		grad_input_desc,
		grad_weight_desc,
		grad_bias_desc,
		input_desc,
		grad_output_desc,
		weight_desc,
		running_mean_desc,
		running_var_desc
    );
    CHECK_RESULT(result);
    const BatchNormBackwardInfo &info = result.take();
    
    *desc_ptr = new Descriptor(
        dtype, std::move(info), WorkSpaceSize,
        nullptr,
        handle->device, handle->device_id
    );    

    return INFINI_STATUS_SUCCESS;
}

#define CALCULATE_BATCH_NORM_BACKWARD(TDATA) \
    CHECK_STATUS(calculate_batch_norm_backward<TDATA>(_info, \
(TDATA *)grad_input, (TDATA *)grad_weight, (TDATA *)grad_bias, (const TDATA *)input, (const TDATA *)grad_output, (const TDATA *)weight, (const TDATA *)running_mean, (const TDATA *)running_var))

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
	void * grad_input,
	void * grad_weight,
	void * grad_bias,
	const void * input,
	const void * grad_output,
	const void * weight,
	const void * running_mean,
	const void * running_var,
    void *stream
) const {
    if (_info.dtype == INFINI_DTYPE_F16) {
        CALCULATE_BATCH_NORM_BACKWARD(fp16_t);
    } else if (_info.dtype == INFINI_DTYPE_BF16) {
        CALCULATE_BATCH_NORM_BACKWARD(bf16_t);
    } else if (_info.dtype == INFINI_DTYPE_F32) {
        CALCULATE_BATCH_NORM_BACKWARD(float);
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    return INFINI_STATUS_SUCCESS;
}
}
