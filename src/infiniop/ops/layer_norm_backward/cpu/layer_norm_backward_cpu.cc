#include "layer_norm_backward_cpu.h"
#include "../../../handle.h"
#include "../../../tensor.h"
#include "../../../../utils.h"
#include "../../../../utils/custom_types.h"
#include <cmath>
#include <cstring>
#include <vector>
#include <type_traits>
#include <cfenv>
#include <algorithm>

namespace op::layer_norm_backward::cpu {

// 设置全局舍入模式为PyTorch默认模式
void setOptimalRoundingMode() {
    fesetround(FE_TONEAREST); // 四舍五入到最近值
}

// 直接从long double转换到目标类型，避免中间float损耗
template<typename T>
T directCast(long double val) {
    if constexpr (std::is_same<T, bf16_t>::value) {
        // BF16数值范围限制
        val = std::max(static_cast<long double>(-65504.0), std::min(static_cast<long double>(65504.0), val));
        return utils::cast<bf16_t>(static_cast<float>(val));
    } else if constexpr (std::is_same<T, fp16_t>::value) {
        // F16数值范围限制
        val = std::max(static_cast<long double>(-65504.0), std::min(static_cast<long double>(65504.0), val));
        return utils::cast<fp16_t>(static_cast<float>(val));
    } else {
        return static_cast<T>(val);
    }
}

// 高精度类型转换
template<typename T>
float preciseCast(const T& val) {
    if constexpr (std::is_same<T, bf16_t>::value) {
        return utils::cast<float>(val);
    } else if constexpr (std::is_same<T, fp16_t>::value) {
        return utils::cast<float>(val);
    } else {
        return static_cast<float>(val);
    }
}

template<typename T>
infiniStatus_t layerNormBackwardImpl(
    void *grad_input,
    void *grad_weight,
    void *grad_bias,
    const void *grad_output,
    const void *input,
    const void *weight,
    const void *input_std_deviation,
    const void *input_standardization,
    const LayerNormBackwardInfo &info) {
    
    // 设置最优舍入模式
    setOptimalRoundingMode();
    
    const T *grad_output_ptr = static_cast<const T *>(grad_output);
    const T *x = static_cast<const T *>(input);
    (void)x; // Suppress unused variable warning
    const T *input_norm = static_cast<const T *>(input_standardization);
    const T *std_dev = static_cast<const T *>(input_std_deviation);
    
    T *grad_input_ptr = static_cast<T *>(grad_input);
    
    const size_t batch_size = info.batch_size();
    const size_t dim = info.dim();
    const bool has_bias = info.has_bias;
    
    // 转换权重到FP32
    std::vector<float> w_float(dim);
    if (info.wtype == INFINI_DTYPE_F32) {
        const float *w_f32 = static_cast<const float *>(weight);
        for (size_t i = 0; i < dim; ++i) {
            w_float[i] = w_f32[i];
        }
    } else {
        const T *w = static_cast<const T *>(weight);
        for (size_t i = 0; i < dim; ++i) {
            w_float[i] = preciseCast(w[i]);
        }
    }
    
    // 初始化grad_weight和grad_bias
    std::vector<float> grad_weight_accum(dim, 0.0f);
    std::vector<float> grad_bias_accum(dim, 0.0f);
    
    // 对每个batch进行反向传播计算
    for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        const T *grad_out_batch = grad_output_ptr + batch_idx * dim;
        const T *input_norm_batch = input_norm + batch_idx * dim;
        T *grad_input_batch = grad_input_ptr + batch_idx * dim;
        
        float std_dev_val = preciseCast(std_dev[batch_idx]);
        // 添加除零保护，避免inf/nan值
        if (std_dev_val <= 1e-8f) {
            std_dev_val = 1e-8f;
        }
        float inv_std_dev = 1.0f / std_dev_val;
        
        // 计算中间变量用于grad_input
        float sum_grad_out = 0.0f;
        float sum_grad_out_norm = 0.0f;
        
        for (size_t i = 0; i < dim; ++i) {
            float grad_out = preciseCast(grad_out_batch[i]);
            float w = w_float[i];
            float input_norm_val = preciseCast(input_norm_batch[i]);
            
            float grad_out_w = grad_out * w;
            sum_grad_out += grad_out_w;
            sum_grad_out_norm += grad_out_w * input_norm_val;
            
            // 累积grad_weight
            grad_weight_accum[i] += grad_out * input_norm_val;
            
            // 累积grad_bias
            if (has_bias) {
                grad_bias_accum[i] += grad_out;
            }
        }
        
        // 计算grad_input
        float mean_grad_out = sum_grad_out / static_cast<float>(dim);
        float mean_grad_out_norm = sum_grad_out_norm / static_cast<float>(dim);
        
        for (size_t i = 0; i < dim; ++i) {
            float grad_out = preciseCast(grad_out_batch[i]);
            float w = w_float[i];
            float input_norm_val = preciseCast(input_norm_batch[i]);
            
            // LayerNorm反向传播公式: grad_input = inv_std * (grad_out * w - mean_grad_out - input_norm * mean_grad_out_norm)
            long double grad_input_val = static_cast<long double>(inv_std_dev) * 
                (grad_out * w - mean_grad_out - input_norm_val * mean_grad_out_norm);
            
            grad_input_batch[i] = directCast<T>(grad_input_val);
        }
    }
    
    // 写入grad_weight
    if (info.wtype == INFINI_DTYPE_F32) {
        float *grad_w_f32 = static_cast<float *>(grad_weight);
        for (size_t i = 0; i < dim; ++i) {
            grad_w_f32[i] = grad_weight_accum[i];
        }
    } else {
        T *grad_w = static_cast<T *>(grad_weight);
        for (size_t i = 0; i < dim; ++i) {
            grad_w[i] = directCast<T>(static_cast<long double>(grad_weight_accum[i]));
        }
    }
    
    // 写入grad_bias
    if (has_bias) {
        if (info.btype == INFINI_DTYPE_F32) {
            float *grad_b_f32 = static_cast<float *>(grad_bias);
            for (size_t i = 0; i < dim; ++i) {
                grad_b_f32[i] = grad_bias_accum[i];
            }
        } else {
            T *grad_b = static_cast<T *>(grad_bias);
            for (size_t i = 0; i < dim; ++i) {
                grad_b[i] = directCast<T>(static_cast<long double>(grad_bias_accum[i]));
            }
        }
    }
    
    return INFINI_STATUS_SUCCESS;
}



infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t grad_input_desc,
    infiniopTensorDescriptor_t grad_weight_desc,
    infiniopTensorDescriptor_t grad_bias_desc,
    infiniopTensorDescriptor_t grad_output_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t weight_desc,
    infiniopTensorDescriptor_t input_std_deviation_desc,
    infiniopTensorDescriptor_t input_standardization_desc,
    float eps) {
    
    if (!handle || !desc_ptr) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    auto result = LayerNormBackwardInfo::create(
        grad_input_desc, grad_weight_desc, grad_bias_desc,
        grad_output_desc, input_desc, weight_desc,
        input_std_deviation_desc, input_standardization_desc, eps);
    
    if (!result) {
        return result.status();
    }
    
    auto info = result.take();
    *desc_ptr = new Descriptor(handle->device, handle->device_id, std::move(info));
    
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::get_workspace_size(size_t *size) const {
    if (!size) {
        return INFINI_STATUS_BAD_PARAM;
    }
    *size = 0; // CPU实现不需要额外workspace
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *grad_input,
    void *grad_weight,
    void *grad_bias,
    const void *grad_output,
    const void *input,
    const void *weight,
    const void *input_std_deviation,
    const void *input_standardization,
    void *stream) const {
    
    if (!grad_input || !grad_weight || !grad_output || !input || !weight ||
        !input_std_deviation || !input_standardization) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    if (info.has_bias && !grad_bias) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    // 根据数据类型调用相应的实现
    switch (info.atype) {
        case INFINI_DTYPE_F32:
            return layerNormBackwardImpl<float>(
                grad_input, grad_weight, grad_bias, grad_output,
                input, weight, input_std_deviation, input_standardization, info);
        case INFINI_DTYPE_F16:
            return layerNormBackwardImpl<fp16_t>(
                grad_input, grad_weight, grad_bias, grad_output,
                input, weight, input_std_deviation, input_standardization, info);
        case INFINI_DTYPE_BF16:
            return layerNormBackwardImpl<bf16_t>(
                grad_input, grad_weight, grad_bias, grad_output,
                input, weight, input_std_deviation, input_standardization, info);
        default:
            return INFINI_STATUS_NOT_IMPLEMENTED;
    }
}

} // namespace op::layer_norm_backward::cpu