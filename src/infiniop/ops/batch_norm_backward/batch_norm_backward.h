#ifndef __BATCH_NORM_BACKWARD_H__
#define __BATCH_NORM_BACKWARD_H__

#include "../../operator.h"
#include "../../tensor.h"
#include "../../../utils/result.hpp"
#include "infinicore.h"

namespace op::batch_norm_backward {

struct BatchNormBackwardInfo {
    size_t _batch_size;         // 批次大小
    size_t _channels;           // 通道数
    size_t _spatial_size;       // 空间维度大小 (H * W)
    size_t total_elements;      // 总元素数量
    size_t input_size;
    size_t output_size;
    
    infiniDtype_t dtype;        // 主数据类型
    infiniDtype_t wtype;        // 权重数据类型
    infiniDtype_t btype;        // 偏置数据类型
    infiniDtype_t atype;        // 激活数据类型
    float momentum;             // 动量参数
    float eps;                  // epsilon值
    bool has_bias;              // 是否有bias参数
    
    std::vector<size_t> grad_input_shape;
    std::vector<size_t> grad_weight_shape;
    std::vector<size_t> grad_bias_shape;
    std::vector<size_t> grad_output_shape;
    std::vector<size_t> input_shape;
    std::vector<size_t> weight_shape;
    std::vector<size_t> running_mean_shape;
    std::vector<size_t> running_var_shape;
    std::vector<size_t> shape;  // 输出形状（兼容参考实现）
    
    std::vector<ptrdiff_t> grad_input_strides;
    std::vector<ptrdiff_t> grad_weight_strides;
    std::vector<ptrdiff_t> grad_bias_strides;
    std::vector<ptrdiff_t> grad_output_strides;
    std::vector<ptrdiff_t> input_strides;
    std::vector<ptrdiff_t> weight_strides;
    std::vector<ptrdiff_t> running_mean_strides;
    std::vector<ptrdiff_t> running_var_strides;
    
    // 兼容参考实现的方法
    size_t ndim() const { return shape.size(); }
    size_t channels() const { return _channels; }
    size_t batch_size() const {
        return _batch_size;
    }
    size_t spatial_size() const {
        return _spatial_size;
    }
    
    static utils::Result<BatchNormBackwardInfo> create(
        infiniopTensorDescriptor_t grad_input_desc,
        infiniopTensorDescriptor_t grad_weight_desc,
        infiniopTensorDescriptor_t grad_bias_desc,
        infiniopTensorDescriptor_t grad_output_desc,
        infiniopTensorDescriptor_t input_desc,
        infiniopTensorDescriptor_t weight_desc,
        infiniopTensorDescriptor_t running_mean_desc,
        infiniopTensorDescriptor_t running_var_desc,
        float momentum,
        float epsilon) {
        
        BatchNormBackwardInfo info;
        
        // 验证输入参数
        if (!grad_input_desc || !grad_output_desc || !input_desc || 
            !weight_desc || !running_mean_desc || !running_var_desc) {
            return INFINI_STATUS_BAD_PARAM;
        }
        
        if (momentum < 0.0f || momentum > 1.0f || epsilon <= 0.0f) {
            return INFINI_STATUS_BAD_PARAM;
        }
        
        // 设置基本信息
        info.dtype = input_desc->dtype();
        info.atype = input_desc->dtype();
        info.wtype = weight_desc->dtype();
        info.momentum = momentum;
        info.eps = epsilon;
        
        // 获取形状信息
        info.input_shape = input_desc->shape();
        info.shape = info.input_shape;
        
        // 计算批次大小、通道数和空间大小
        if (info.input_shape.size() < 2) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        
        info._batch_size = info.input_shape[0];
        info._channels = info.input_shape[1];
        info._spatial_size = 1;
        for (size_t i = 2; i < info.input_shape.size(); ++i) {
            info._spatial_size *= info.input_shape[i];
        }
        
        info.total_elements = info._batch_size * info._channels * info._spatial_size;
        info.input_size = info.total_elements;
        info.output_size = info.total_elements;
        
        // 设置各个张量的形状和步长
        info.grad_input_shape = grad_input_desc->shape();
        info.grad_input_strides = grad_input_desc->strides();
        
        if (grad_weight_desc) {
            info.grad_weight_shape = grad_weight_desc->shape();
            info.grad_weight_strides = grad_weight_desc->strides();
        }
        
        if (grad_bias_desc) {
            info.grad_bias_shape = grad_bias_desc->shape();
            info.grad_bias_strides = grad_bias_desc->strides();
            info.btype = grad_bias_desc->dtype();
            info.has_bias = true;
        } else {
            info.has_bias = false;
        }
        
        info.grad_output_shape = grad_output_desc->shape();
        info.grad_output_strides = grad_output_desc->strides();
        
        info.input_strides = input_desc->strides();
        
        if (weight_desc) {
            info.weight_shape = weight_desc->shape();
            info.weight_strides = weight_desc->strides();
        }
        
        if (running_mean_desc) {
            info.running_mean_shape = running_mean_desc->shape();
            info.running_mean_strides = running_mean_desc->strides();
        }
        
        if (running_var_desc) {
            info.running_var_shape = running_var_desc->shape();
            info.running_var_strides = running_var_desc->strides();
        }
        
        return utils::Result<BatchNormBackwardInfo>(info);
    }
};

struct Descriptor {
    BatchNormBackwardInfo info;
    infiniDevice_t device;
    
    static utils::Result<Descriptor> create(
        infiniopTensorDescriptor_t grad_input_desc,
        infiniopTensorDescriptor_t grad_weight_desc,
        infiniopTensorDescriptor_t grad_bias_desc,
        infiniopTensorDescriptor_t grad_output_desc,
        infiniopTensorDescriptor_t input_desc,
        infiniopTensorDescriptor_t weight_desc,
        infiniopTensorDescriptor_t running_mean_desc,
        infiniopTensorDescriptor_t running_var_desc,
        float momentum,
        float epsilon);
    
    utils::Result<size_t> get_workspace_size() const;
    
    infiniStatus_t calculate(
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
        void *stream) const;
};

} // namespace op::batch_norm_backward

#endif // __BATCH_NORM_BACKWARD_H__