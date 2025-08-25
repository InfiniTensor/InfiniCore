#include "batch_norm_cpu.h"
#include "../../../handle.h"
#include "../../../tensor.h"
#include "../../../../utils.h"
#include "../../../../utils/custom_types.h"
#include <cmath>
#include <cstring>

namespace op::batch_norm::cpu {

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t weight_desc,
    infiniopTensorDescriptor_t bias_desc,
    infiniopTensorDescriptor_t running_mean_desc,
    infiniopTensorDescriptor_t running_var_desc,
    float momentum,
    float eps) {
    
    // 验证输入参数
    if (!handle || !desc_ptr || !output_desc || !input_desc || 
        !weight_desc || !bias_desc || !running_mean_desc || !running_var_desc) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    if (momentum < 0.0f || momentum > 1.0f || eps <= 0.0f) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    // 检查数据类型一致性
    if (input_desc->dtype() != output_desc->dtype() ||
        input_desc->dtype() != weight_desc->dtype() ||
        input_desc->dtype() != bias_desc->dtype() ||
        input_desc->dtype() != running_mean_desc->dtype() ||
        input_desc->dtype() != running_var_desc->dtype()) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    // 检查输入维度
    if (input_desc->ndim() < 2) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }
    
    // 检查形状一致性
    if (input_desc->ndim() != output_desc->ndim()) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }
    
    for (size_t i = 0; i < input_desc->ndim(); ++i) {
        if (input_desc->dim(i) != output_desc->dim(i)) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
    }
    
    // weight, bias, running_mean, running_var 应该是1D张量，长度为channels
    size_t channels = input_desc->dim(1);
    if (weight_desc->ndim() != 1 || weight_desc->dim(0) != channels ||
        bias_desc->ndim() != 1 || bias_desc->dim(0) != channels ||
        running_mean_desc->ndim() != 1 || running_mean_desc->dim(0) != channels ||
        running_var_desc->ndim() != 1 || running_var_desc->dim(0) != channels) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    auto desc = new Descriptor();
    desc->device_type = handle->device;
    
    // 填充BatchNormInfo
    desc->info.batch_size = input_desc->dim(0);
    desc->info.channels = channels;
    
    // 计算spatial_size (除了batch和channel维度的所有维度的乘积)
    desc->info.spatial_size = 1;
    for (size_t i = 2; i < input_desc->ndim(); ++i) {
        desc->info.spatial_size *= input_desc->dim(i);
    }
    
    desc->info.input_size = input_desc->numel();
    desc->info.output_size = output_desc->numel();
    desc->info.dtype = input_desc->dtype();
    desc->info.momentum = momentum;
    desc->info.eps = eps;
    
    // 复制形状和步长信息
    desc->info.input_shape = input_desc->shape();
    desc->info.output_shape = output_desc->shape();
    desc->info.input_strides = input_desc->strides();
    desc->info.output_strides = output_desc->strides();
    desc->info.weight_strides = weight_desc->strides();
    desc->info.bias_strides = bias_desc->strides();
    desc->info.running_mean_strides = running_mean_desc->strides();
    desc->info.running_var_strides = running_var_desc->strides();
    
    *desc_ptr = desc;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::get_workspace_size(size_t *size) const {
    if (!size) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    // 需要临时存储每个channel的均值和方差
    size_t dtype_size = infiniSizeOf(info.dtype);
    *size = 2 * info.channels * dtype_size; // mean + var
    
    return INFINI_STATUS_SUCCESS;
}

template<typename T>
static void batch_norm_impl(
    T *output,
    const T *input,
    const T *weight,
    const T *bias,
    T *running_mean,
    T *running_var,
    T *workspace_mean,
    T *workspace_var,
    const BatchNormInfo &info) {
    
    const size_t batch_size = info.batch_size;
    const size_t channels = info.channels;
    const size_t spatial_size = info.spatial_size;
    const float momentum = info.momentum;
    const float eps = info.eps;
    const float one_minus_momentum = 1.0f - info.momentum;
    
    // 计算每个channel的均值
    for (size_t c = 0; c < channels; ++c) {
        float sum = 0.0f;
        for (size_t n = 0; n < batch_size; ++n) {
            for (size_t s = 0; s < spatial_size; ++s) {
                size_t idx = n * channels * spatial_size + c * spatial_size + s;
                if constexpr (std::is_same<T, fp16_t>::value || std::is_same<T, bf16_t>::value) {
                    sum += utils::cast<float>(input[idx]);
                } else {
                    sum += input[idx];
                }
            }
        }
        float mean_val = sum / static_cast<float>(batch_size * spatial_size);
        if constexpr (std::is_same<T, fp16_t>::value || std::is_same<T, bf16_t>::value) {
            workspace_mean[c] = utils::cast<T>(mean_val);
        } else {
            workspace_mean[c] = static_cast<T>(mean_val);
        }
    }
    
    // 计算每个channel的方差
    for (size_t c = 0; c < channels; ++c) {
        float sum_sq_diff = 0.0f;
        float mean_val;
        if constexpr (std::is_same<T, fp16_t>::value || std::is_same<T, bf16_t>::value) {
            mean_val = utils::cast<float>(workspace_mean[c]);
        } else {
            mean_val = workspace_mean[c];
        }
        for (size_t n = 0; n < batch_size; ++n) {
            for (size_t s = 0; s < spatial_size; ++s) {
                size_t idx = n * channels * spatial_size + c * spatial_size + s;
                float input_val;
                if constexpr (std::is_same<T, fp16_t>::value || std::is_same<T, bf16_t>::value) {
                    input_val = utils::cast<float>(input[idx]);
                } else {
                    input_val = input[idx];
                }
                float diff = input_val - mean_val;
                sum_sq_diff += diff * diff;
            }
        }
        float var_val = sum_sq_diff / static_cast<float>(batch_size * spatial_size);
        if constexpr (std::is_same<T, fp16_t>::value || std::is_same<T, bf16_t>::value) {
            workspace_var[c] = utils::cast<T>(var_val);
        } else {
            workspace_var[c] = static_cast<T>(var_val);
        }
    }
    
    // 更新running statistics
    for (size_t c = 0; c < channels; ++c) {
        float old_mean, old_var, new_mean, new_var;
        if constexpr (std::is_same<T, fp16_t>::value || std::is_same<T, bf16_t>::value) {
            old_mean = utils::cast<float>(running_mean[c]);
            old_var = utils::cast<float>(running_var[c]);
            new_mean = utils::cast<float>(workspace_mean[c]);
            new_var = utils::cast<float>(workspace_var[c]);
            running_mean[c] = utils::cast<T>(momentum * new_mean + one_minus_momentum * old_mean);
            running_var[c] = utils::cast<T>(momentum * new_var + one_minus_momentum * old_var);
        } else {
            old_mean = running_mean[c];
            old_var = running_var[c];
            new_mean = workspace_mean[c];
            new_var = workspace_var[c];
            running_mean[c] = static_cast<T>(momentum * new_mean + one_minus_momentum * old_mean);
            running_var[c] = static_cast<T>(momentum * new_var + one_minus_momentum * old_var);
        }
    }
    
    // 执行归一化和线性变换
    for (size_t n = 0; n < batch_size; ++n) {
        for (size_t c = 0; c < channels; ++c) {
            float mean_val, var_val, w_val, b_val;
            if constexpr (std::is_same<T, fp16_t>::value || std::is_same<T, bf16_t>::value) {
                mean_val = utils::cast<float>(workspace_mean[c]);
                var_val = utils::cast<float>(workspace_var[c]);
                w_val = utils::cast<float>(weight[c]);
                b_val = utils::cast<float>(bias[c]);
            } else {
                mean_val = workspace_mean[c];
                var_val = workspace_var[c];
                w_val = weight[c];
                b_val = bias[c];
            }
            float std_inv = 1.0f / std::sqrt(var_val + eps);
            
            for (size_t s = 0; s < spatial_size; ++s) {
                size_t idx = n * channels * spatial_size + c * spatial_size + s;
                float input_val;
                if constexpr (std::is_same<T, fp16_t>::value || std::is_same<T, bf16_t>::value) {
                    input_val = utils::cast<float>(input[idx]);
                } else {
                    input_val = input[idx];
                }
                float normalized = (input_val - mean_val) * std_inv;
                float result = normalized * w_val + b_val;
                if constexpr (std::is_same<T, fp16_t>::value || std::is_same<T, bf16_t>::value) {
                    output[idx] = utils::cast<T>(result);
                } else {
                    output[idx] = static_cast<T>(result);
                }
            }
        }
    }
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *output,
    const void *input,
    const void *weight,
    const void *bias,
    void *running_mean,
    void *running_var,
    void *stream) const {
    
    if (!output || !input || !weight || !bias || !running_mean || !running_var) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    size_t required_workspace_size;
    auto status = get_workspace_size(&required_workspace_size);
    if (status != INFINI_STATUS_SUCCESS) {
        return status;
    }
    
    if (workspace_size < required_workspace_size) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    size_t dtype_size = infiniSizeOf(info.dtype);
    void *workspace_mean = workspace;
    void *workspace_var = static_cast<char*>(workspace) + info.channels * dtype_size;
    
    switch (info.dtype) {
        case INFINI_DTYPE_F32:
            batch_norm_impl<float>(
                static_cast<float*>(output),
                static_cast<const float*>(input),
                static_cast<const float*>(weight),
                static_cast<const float*>(bias),
                static_cast<float*>(running_mean),
                static_cast<float*>(running_var),
                static_cast<float*>(workspace_mean),
                static_cast<float*>(workspace_var),
                info);
            break;
        case INFINI_DTYPE_F16:
            batch_norm_impl<fp16_t>(
                static_cast<fp16_t*>(output),
                static_cast<const fp16_t*>(input),
                static_cast<const fp16_t*>(weight),
                static_cast<const fp16_t*>(bias),
                static_cast<fp16_t*>(running_mean),
                static_cast<fp16_t*>(running_var),
                static_cast<fp16_t*>(workspace_mean),
                static_cast<fp16_t*>(workspace_var),
                info);
            break;
        case INFINI_DTYPE_BF16:
            batch_norm_impl<bf16_t>(
                static_cast<bf16_t*>(output),
                static_cast<const bf16_t*>(input),
                static_cast<const bf16_t*>(weight),
                static_cast<const bf16_t*>(bias),
                static_cast<bf16_t*>(running_mean),
                static_cast<bf16_t*>(running_var),
                static_cast<bf16_t*>(workspace_mean),
                static_cast<bf16_t*>(workspace_var),
                info);
            break;
        default:
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::batch_norm::cpu