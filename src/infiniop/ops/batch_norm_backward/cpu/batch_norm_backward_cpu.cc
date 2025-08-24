#include "batch_norm_backward_cpu.h"
#include "../../../handle.h"
#include "../../../tensor.h"
#include "../../../../utils.h"
#include "../../../../utils/custom_types.h"
#include <cmath>
#include <cstring>
#include <vector>
#include <type_traits>
#include <algorithm>

namespace op::batch_norm_backward::cpu {

// 超高精度类型转换
template<typename T>
inline long double ultraPreciseCast(const T& val) {
    if constexpr (std::is_same_v<T, fp16_t>) {
        return static_cast<long double>(utils::cast<float>(val));
    } else if constexpr (std::is_same_v<T, bf16_t>) {
        return static_cast<long double>(utils::cast<float>(val));
    } else {
        return static_cast<long double>(val);
    }
}

// 高精度类型转换函数
template<typename T>
inline float preciseCast(const T& val) {
    if constexpr (std::is_same_v<T, fp16_t>) {
        return utils::cast<float>(val);
    } else if constexpr (std::is_same_v<T, bf16_t>) {
        return utils::cast<float>(val);
    } else {
        return static_cast<float>(val);
    }
}

template<typename T>
inline T directCast(long double val) {
    if constexpr (std::is_same_v<T, fp16_t>) {
        return utils::cast<fp16_t>(static_cast<float>(val));
    } else if constexpr (std::is_same_v<T, bf16_t>) {
        return utils::cast<bf16_t>(static_cast<float>(val));
    } else {
        return static_cast<T>(val);
    }
}

// Kahan求和算法，提高数值精度
struct EnhancedKahanSum {
    long double sum = 0.0L;
    long double c = 0.0L; // 补偿项
    
    void add(long double value) {
        long double y = value - c;
        long double t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    
    long double get() const {
        return sum;
    }
};

// 数值清洗函数
inline long double sanitizeValue(long double val) {
    if (std::isnan(val) || std::isinf(val)) {
        return 0.0L;
    }
    return val;
}

// BatchNorm反向传播核心实现（优化版本）
// 在推理模式下，使用固定的running_mean和running_var
template<typename T>
void batch_norm_backward_impl(
    T* grad_input,
    T* grad_weight, 
    T* grad_bias,
    const T* grad_output,
    const T* input,
    const T* weight,
    const T* running_mean,
    const T* running_var,
    const BatchNormBackwardInfo& info) {
    
    const size_t batch_size = info.batch_size();
    const size_t channels = info.channels();
    const size_t spatial_size = info.spatial_size();
    const long double eps = static_cast<long double>(info.eps);
    
    // 对每个通道并行处理
    #pragma omp parallel for
    for (size_t c = 0; c < channels; ++c) {
        // 使用running_mean和running_var（推理模式的关键）
        long double mean_val = ultraPreciseCast(running_mean[c]);
        long double var_val = ultraPreciseCast(running_var[c]);
        long double weight_val = ultraPreciseCast(weight[c]);
        
        // 添加除零保护，避免inf/nan值
        if (var_val <= 1e-12L) {
            var_val = 1e-12L;
        }
        
        // 计算标准差的倒数（使用long double提高精度）
        long double variance = var_val + eps;
        long double inv_std = 1.0L / std::sqrt(variance);
        
        // 数值清洗
        inv_std = sanitizeValue(inv_std);
        
        // 计算grad_bias (如果需要) - 使用Kahan求和
        if (grad_bias) {
            EnhancedKahanSum bias_grad_sum;
            for (size_t n = 0; n < batch_size; ++n) {
                for (size_t s = 0; s < spatial_size; ++s) {
                    size_t idx = n * channels * spatial_size + c * spatial_size + s;
                    long double grad_out_val = ultraPreciseCast(grad_output[idx]);
                    bias_grad_sum.add(grad_out_val);
                }
            }
            grad_bias[c] = directCast<T>(sanitizeValue(bias_grad_sum.get()));
        }
        
        // 计算grad_weight (如果需要) - 使用Kahan求和
        if (grad_weight) {
            EnhancedKahanSum weight_grad_sum;
            for (size_t n = 0; n < batch_size; ++n) {
                for (size_t s = 0; s < spatial_size; ++s) {
                    size_t idx = n * channels * spatial_size + c * spatial_size + s;
                    long double input_val = ultraPreciseCast(input[idx]);
                    long double grad_out_val = ultraPreciseCast(grad_output[idx]);
                    
                    // 标准化的输入值（使用long double精度）
                    long double normalized = (input_val - mean_val) * inv_std;
                    long double contribution = grad_out_val * normalized;
                    weight_grad_sum.add(sanitizeValue(contribution));
                }
            }
            grad_weight[c] = directCast<T>(sanitizeValue(weight_grad_sum.get()));
        }
        
        // 计算grad_input
        if (grad_input) {
            for (size_t n = 0; n < batch_size; ++n) {
                for (size_t s = 0; s < spatial_size; ++s) {
                    size_t idx = n * channels * spatial_size + c * spatial_size + s;
                    long double grad_out_val = ultraPreciseCast(grad_output[idx]);
                    
                    // 在推理模式下，grad_input的计算简化为：
                    // grad_input = grad_output * weight * inv_std
                    long double grad_in_val = grad_out_val * weight_val * inv_std;
                    grad_input[idx] = directCast<T>(sanitizeValue(grad_in_val));
                }
            }
        }
    }
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
    infiniopTensorDescriptor_t running_mean_desc,
    infiniopTensorDescriptor_t running_var_desc,
    float momentum,
    float eps) {
    
    // 验证输入参数
    if (!handle || !desc_ptr || !grad_output_desc || !input_desc || 
        !weight_desc || !running_mean_desc || !running_var_desc) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    if (eps <= 0.0f) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    // 检查数据类型一致性
    if (input_desc->dtype() != grad_output_desc->dtype() ||
        input_desc->dtype() != weight_desc->dtype() ||
        input_desc->dtype() != running_mean_desc->dtype() ||
        input_desc->dtype() != running_var_desc->dtype()) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    if (grad_input_desc && grad_input_desc->dtype() != input_desc->dtype()) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    if (grad_weight_desc && grad_weight_desc->dtype() != weight_desc->dtype()) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    if (grad_bias_desc && grad_bias_desc->dtype() != weight_desc->dtype()) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    // 检查输入维度
    if (input_desc->ndim() < 2) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }
    
    // 创建BatchNormBackwardInfo
    auto info_result = BatchNormBackwardInfo::create(
        grad_input_desc, grad_weight_desc, grad_bias_desc, grad_output_desc,
        input_desc, weight_desc, running_mean_desc, running_var_desc,
        momentum, eps);
    
    if (!info_result) {
        return info_result.status();
    }
    
    auto info = info_result.take();
    
    // 创建Descriptor
    auto desc = new Descriptor(handle->device, handle->device_id, std::move(info));
    *desc_ptr = desc;
    
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::get_workspace_size(size_t *size) const {
    if (!size) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    // BatchNorm反向传播不需要额外的workspace
    *size = 0;
    
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *grad_input,
    void *grad_weight,
    void *grad_bias,
    const void *grad_output,
    const void *input,
    const void *weight,
    const void *running_mean,
    const void *running_var,
    void *stream) const {
    
    // 验证必需的输入参数
    if (!grad_output || !input || !weight || !running_mean || !running_var) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    // 根据数据类型调用相应的实现
    switch (info.dtype) {
        case INFINI_DTYPE_F32:
            batch_norm_backward_impl<float>(
                static_cast<float*>(grad_input),
                static_cast<float*>(grad_weight),
                static_cast<float*>(grad_bias),
                static_cast<const float*>(grad_output),
                static_cast<const float*>(input),
                static_cast<const float*>(weight),
                static_cast<const float*>(running_mean),
                static_cast<const float*>(running_var),
                info);
            break;
        case INFINI_DTYPE_F16:
            batch_norm_backward_impl<fp16_t>(
                static_cast<fp16_t*>(grad_input),
                static_cast<fp16_t*>(grad_weight),
                static_cast<fp16_t*>(grad_bias),
                static_cast<const fp16_t*>(grad_output),
                static_cast<const fp16_t*>(input),
                static_cast<const fp16_t*>(weight),
                static_cast<const fp16_t*>(running_mean),
                static_cast<const fp16_t*>(running_var),
                info);
            break;
        case INFINI_DTYPE_BF16:
            batch_norm_backward_impl<bf16_t>(
                static_cast<bf16_t*>(grad_input),
                static_cast<bf16_t*>(grad_weight),
                static_cast<bf16_t*>(grad_bias),
                static_cast<const bf16_t*>(grad_output),
                static_cast<const bf16_t*>(input),
                static_cast<const bf16_t*>(weight),
                static_cast<const bf16_t*>(running_mean),
                static_cast<const bf16_t*>(running_var),
                info);
            break;
        default:
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::batch_norm_backward::cpu