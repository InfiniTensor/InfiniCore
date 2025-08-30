#include "layer_norm_backward_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../../reduce/cpu/reduce.h"
#include "../info.h"

namespace op::layer_norm_backward::cpu {

template <typename Tdata>
infiniStatus_t calculate_layer_norm_backward(
    const LayerNormBackwardInfo &info,
    Tdata * grad_input,
    Tdata * grad_weight,
    Tdata * grad_bias,
    const Tdata * grad_output,
    const Tdata * weight,
    const Tdata * input_standardization,
    const Tdata * input_std_deviation
) {
//  -------------------------------- start: perform operator on CPU --------------------------------
    #pragma omp parallel for
    for (size_t i = 0; i < info.feature_size; i ++) {
        auto grad_output_ptr = grad_output + i;
        auto input_standard_ptr = input_standardization + i;
        float sum_dy = 0.;
        float sum_dy_norm_x = 0.;
        for (size_t b = 0; b < info.batch_size; b++) {

            for (size_t c = 0; c < info.channel_size; c ++) {
                float dy = utils::cast<float>(*(grad_output_ptr + b * info.grad_output_strides[0] + c * info.grad_output_strides[1]));
                float norm_x = utils::cast<float>(*(input_standard_ptr + b * info.input_standardization_strides[0] + c * info.input_standardization_strides[1]));
                float dy_norm_x = dy * norm_x;
                sum_dy += dy;
                sum_dy_norm_x += dy_norm_x;

                
            }
        }
        *(grad_weight + i * info.grad_weight_strides[0]) = utils::cast<Tdata>(sum_dy_norm_x);
        if (info.bias)
            *(grad_bias + i * info.grad_bias_strides[0]) = utils::cast<Tdata>(sum_dy);
    }
    #pragma omp barrier

    #pragma omp parallel for
    for (size_t n = 0; n < info.batch_size * info.channel_size; n ++) {
        size_t b = n / info.channel_size, c = n % info.channel_size;
            
        float std = utils::cast<float>(*(input_std_deviation + b * info.input_std_deviation_strides[0] + c * info.input_std_deviation_strides[1]));
        
        float sum_dy_w = 0;
        float sum_dy_w_norm_x = 0;
        auto grad_output_ptr = grad_output + b * info.grad_output_strides[0] + c * info.grad_output_strides[1];
        auto input_standard_ptr = input_standardization + b * info.input_standardization_strides[0] + c * info.input_standardization_strides[1];
        auto grad_input_ptr = grad_input + b * info.grad_input_strides[0] + c * info.grad_input_strides[1];

        for (size_t i = 0; i < info.feature_size; i ++) {
            float wt = utils::cast<float>(*(weight + i * info.weight_strides[0]));
            float dy_w = utils::cast<float>(grad_output_ptr[i]) * wt;
            float norm_x = utils::cast<float>(input_standard_ptr[i]);
            sum_dy_w += dy_w;
            sum_dy_w_norm_x += dy_w * norm_x;
        }
        for (size_t i = 0; i < info.feature_size; i ++) {
            float wt = utils::cast<float>(*(weight + i * info.weight_strides[0]));
            float dy = utils::cast<float>(grad_output_ptr[i]);
            float norm_x = utils::cast<float>(input_standard_ptr[i]);
            grad_input_ptr[i] = utils::cast<Tdata>(wt * dy / std + (
                - sum_dy_w - norm_x * sum_dy_w_norm_x
            ) / (std * info.feature_size));
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
    infiniopTensorDescriptor_t grad_output_desc,
    infiniopTensorDescriptor_t weight_desc,
    infiniopTensorDescriptor_t input_standardization_desc,
    infiniopTensorDescriptor_t input_std_deviation_desc
) {
    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);

//  --------------------- start: check data type and calculate workspace size ----------------------
    auto dtype = grad_input_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);
    size_t WorkSpaceSize = 0;
//  ---------------------- end: check data type and calculate workspace size -----------------------

    auto result = LayerNormBackwardInfo::createLayerNormBackwardInfo(
        grad_input_desc,
        grad_weight_desc,
        grad_bias_desc,
        grad_output_desc,
        weight_desc,
        input_standardization_desc,
        input_std_deviation_desc
    );
    CHECK_RESULT(result);
    const LayerNormBackwardInfo &info = result.take();
    
    *desc_ptr = new Descriptor(
        dtype, std::move(info), WorkSpaceSize,
        nullptr,
        handle->device, handle->device_id
    );    

    return INFINI_STATUS_SUCCESS;
}


infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void * grad_input,
    void * grad_weight,
    void * grad_bias,
    const void * grad_output,
    const void * weight,
    const void * input_standardization,
    const void * input_std_deviation,
    void *stream
) const {
    #define CALCULATE_LAYER_NORM_BACKWARD(TDATA) \
        CHECK_STATUS(calculate_layer_norm_backward<TDATA>(_info, \
    (TDATA *)grad_input, (TDATA *)grad_weight, (TDATA *)grad_bias, (const TDATA *)grad_output, (const TDATA *)weight, (const TDATA *)input_standardization, (const TDATA *)input_std_deviation))

    if (_info.dtype == INFINI_DTYPE_F16) {
        CALCULATE_LAYER_NORM_BACKWARD(fp16_t);
    } else if (_info.dtype == INFINI_DTYPE_BF16) {
        CALCULATE_LAYER_NORM_BACKWARD(bf16_t);
    } else if (_info.dtype == INFINI_DTYPE_F32) {
        CALCULATE_LAYER_NORM_BACKWARD(float);
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    return INFINI_STATUS_SUCCESS;
}
}
