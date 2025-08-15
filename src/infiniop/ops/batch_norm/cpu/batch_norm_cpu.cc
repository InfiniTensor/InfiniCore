#include "batch_norm_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../../reduce/cpu/reduce.h"

namespace op::batch_norm::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
	infiniopTensorDescriptor_t output_desc,
	infiniopTensorDescriptor_t running_mean_desc,
	infiniopTensorDescriptor_t running_var_desc,
	infiniopTensorDescriptor_t input_desc,
	infiniopTensorDescriptor_t weight_desc,
	infiniopTensorDescriptor_t bias_desc,
	float momentum,
	float eps
) {
    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto dtype = input_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);
    auto result = BatchNormInfo::createBatchNormInfo(
		output_desc,
		running_mean_desc,
		running_var_desc,
		input_desc,
		weight_desc,
		bias_desc,
		momentum,
		eps
    );
    CHECK_RESULT(result);
    const BatchNormInfo &info = result.take();
    size_t WorkSpaceSize = 0;
    *desc_ptr = new Descriptor(
        dtype, std::move(info), WorkSpaceSize,
        nullptr,
        handle->device, handle->device_id
    );    

    return INFINI_STATUS_SUCCESS;
}

template <typename Tdata>
infiniStatus_t calculate_batch_norm(
    const BatchNormInfo &info,
	Tdata * output,
	Tdata * running_mean,
	Tdata * running_var,
	const Tdata *input,
	const Tdata *weight,
	const Tdata *bias
) {

#pragma omp parallel for
    for(size_t c = 0; c < info.channel_size; c++)
    {
        float sum_sq = 0., sum=0.;
        for(size_t b = 0; b < info.batch_size; b++)
        {
            sum += op::common_cpu::reduce_op::sum(
                input + (b * info.channel_size + c) * info.dim_size,
                info.dim_size,
                1
            );
            sum_sq += op::common_cpu::reduce_op::sumSquared(
                input + (b * info.channel_size + c) * info.dim_size,
                info.dim_size,
                1
            );
        }
        float batch_and_dim_size = (info.batch_size * info.dim_size);
        float E = sum / batch_and_dim_size;
        float var_biased = sum_sq / batch_and_dim_size - E * E;
        float var_unbiased = var_biased * batch_and_dim_size / (batch_and_dim_size - 1.0);

        auto running_mean_ptr = running_mean + c * info.running_mean_stride;
        auto running_var_ptr = running_var + c * info.running_var_stride;
        *running_mean_ptr =  utils::cast<Tdata>((1 - info.momentum) * utils::cast<float>(*running_mean_ptr) + info.momentum * E);
        *running_var_ptr =  utils::cast<Tdata>((1 - info.momentum) * utils::cast<float>(*running_var_ptr) + info.momentum * var_unbiased);

        for(size_t b = 0; b < info.batch_size; b++)
        {
            for(size_t d = 0; d < info.dim_size; d++)
            {
                auto input_ptr = input + ((b * info.channel_size + c) * info.dim_size) + d;
                auto output_ptr = output + ((b * info.channel_size + c) * info.dim_size) + d;;
                auto weight_ptr = weight + c * info.weight_stride;
                auto bias_ptr = bias + c * info.bias_stride;
                *output_ptr = utils::cast<Tdata>(
                    (utils::cast<float>(*input_ptr) - E) / std::sqrt(var_biased + info.eps) * utils::cast<float>(*weight_ptr) + utils::cast<float>(*bias_ptr)
                );
            }
        }    
    }
    return INFINI_STATUS_SUCCESS;
}

#define CALCULATE_BATCH_NORM(TDATA) \
    CHECK_STATUS(calculate_batch_norm<TDATA>(_info, \
(TDATA *)output, (TDATA *)running_mean, (TDATA *)running_var, (const TDATA *)input, (const TDATA *)weight, (const TDATA *)bias))

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
	void * output,
	void * running_mean,
	void * running_var,
	const void * input,
	const void * weight,
	const void * bias,
    void *stream
) const {

    if (_info.dtype == INFINI_DTYPE_F16) {
        CALCULATE_BATCH_NORM(fp16_t);
    } else if (_info.dtype == INFINI_DTYPE_BF16) {
        CALCULATE_BATCH_NORM(bf16_t);
    } else if (_info.dtype == INFINI_DTYPE_F32) {
        CALCULATE_BATCH_NORM(float);
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::exp::cpu
