#ifndef __LAYER_NORM_BACKWARD_H__
#define __LAYER_NORM_BACKWARD_H__

#include "../../operator.h"
#include "../../tensor.h"
#include "../../../utils/result.hpp"
#include "infinicore.h"

namespace op::layer_norm_backward {

struct LayerNormBackwardInfo {
    size_t _batch_size;         // 批次大小
    size_t _normalized_size;    // 归一化维度大小，最后一维的大小，用于归一化
    size_t total_elements;      // 总元素数量
    size_t input_size;
    size_t output_size;
    
    infiniDtype_t dtype;        // 主数据类型
    infiniDtype_t wtype;        // 权重数据类型
    infiniDtype_t btype;        // 偏置数据类型
    infiniDtype_t atype;        // 激活数据类型
    float eps;                  // epsilon值
    float epsilon;              // epsilon值（兼容参考实现）
    bool has_bias;              // 是否有bias参数
    
    std::vector<size_t> input_grad_shape;
    std::vector<size_t> weight_grad_shape;
    std::vector<size_t> bias_grad_shape;
    std::vector<size_t> output_grad_shape;
    std::vector<size_t> input_shape;
    std::vector<size_t> weight_shape;
    std::vector<size_t> input_std_deviation_shape;
    std::vector<size_t> input_standardization_shape;
    std::vector<size_t> shape;  // 输出形状（兼容参考实现）
    
    std::vector<ptrdiff_t> input_grad_strides;
    std::vector<ptrdiff_t> weight_grad_strides;
    std::vector<ptrdiff_t> bias_grad_strides;
    std::vector<ptrdiff_t> output_grad_strides;
    std::vector<ptrdiff_t> input_strides;
    std::vector<ptrdiff_t> weight_strides;
    std::vector<ptrdiff_t> input_std_deviation_strides;
    std::vector<ptrdiff_t> input_standardization_strides;
    
    // 兼容参考实现的方法
    size_t ndim() const { return shape.size(); }
    size_t dim() const { return _normalized_size; }
    size_t batch_size() const {
        return _batch_size;
    }
    
    // 创建LayerNormBackwardInfo的静态方法
    static utils::Result<LayerNormBackwardInfo> create(
        infiniopTensorDescriptor_t grad_input_desc,
        infiniopTensorDescriptor_t grad_weight_desc,
        infiniopTensorDescriptor_t grad_bias_desc,
        infiniopTensorDescriptor_t grad_output_desc,
        infiniopTensorDescriptor_t input_desc,
        infiniopTensorDescriptor_t weight_desc,
        infiniopTensorDescriptor_t input_std_deviation_desc,
        infiniopTensorDescriptor_t input_standardization_desc,
        float epsilon) {
        
        // 验证输入参数
        if (!grad_input_desc || !grad_weight_desc || !grad_output_desc || 
            !input_desc || !weight_desc || !input_std_deviation_desc || 
            !input_standardization_desc) {
            return INFINI_STATUS_BAD_PARAM;
        }
        
        if (epsilon <= 0.0f) {
            return INFINI_STATUS_BAD_PARAM;
        }
        
        auto atype = input_desc->dtype();
        auto wtype = weight_desc->dtype();
        bool has_bias = (grad_bias_desc != nullptr);
        auto btype = has_bias ? grad_bias_desc->dtype() : atype;
        
        // 检查数据类型一致性
        if (input_desc->dtype() != grad_output_desc->dtype() ||
            input_desc->dtype() != grad_input_desc->dtype() ||
            input_desc->dtype() != input_std_deviation_desc->dtype() ||
            input_desc->dtype() != input_standardization_desc->dtype()) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        
        // 检查输入维度（至少3D）
        if (input_desc->ndim() < 3) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        
        // 检查形状兼容性
        if (input_desc->ndim() != grad_output_desc->ndim() ||
            input_desc->ndim() != grad_input_desc->ndim() ||
            input_desc->ndim() != input_standardization_desc->ndim()) {
            return INFINI_STATUS_BAD_PARAM;
        }
        
        for (size_t i = 0; i < input_desc->ndim(); ++i) {
            if (input_desc->dim(i) != grad_output_desc->dim(i) ||
                input_desc->dim(i) != grad_input_desc->dim(i) ||
                input_desc->dim(i) != input_standardization_desc->dim(i)) {
                return INFINI_STATUS_BAD_PARAM;
            }
        }
        
        // weight应该是1D张量，长度为最后一维
        size_t normalized_size = input_desc->dim(input_desc->ndim() - 1);
        if (weight_desc->ndim() != 1 || weight_desc->dim(0) != normalized_size ||
            grad_weight_desc->ndim() != 1 || grad_weight_desc->dim(0) != normalized_size) {
            return INFINI_STATUS_BAD_PARAM;
        }
        
        // 检查bias形状（如果存在）
        if (has_bias) {
            if (grad_bias_desc->ndim() != 1 || grad_bias_desc->dim(0) != normalized_size) {
                return INFINI_STATUS_BAD_PARAM;
            }
        }
        
        // 检查input_std_deviation形状（应该是input去掉最后一维）
        if (input_std_deviation_desc->ndim() != input_desc->ndim() - 1) {
            return INFINI_STATUS_BAD_PARAM;
        }
        for (size_t i = 0; i < input_std_deviation_desc->ndim(); ++i) {
            if (input_std_deviation_desc->dim(i) != input_desc->dim(i)) {
                return INFINI_STATUS_BAD_PARAM;
            }
        }
        
        // 创建LayerNormBackwardInfo
        LayerNormBackwardInfo info;
        
        // 计算batch_size（除了最后一维的所有维度的乘积）
        info._batch_size = 1;
        for (size_t i = 0; i < input_desc->ndim() - 1; ++i) {
            info._batch_size *= input_desc->dim(i);
        }
        
        info._normalized_size = normalized_size;
        info.total_elements = input_desc->numel();
        info.input_size = input_desc->numel();
        info.output_size = grad_output_desc->numel();
        info.dtype = atype;
        info.atype = atype;
        info.wtype = wtype;
        info.btype = btype;
        info.eps = epsilon;
        info.epsilon = epsilon;
        info.has_bias = has_bias;
        
        // 复制形状和步长信息
        info.input_grad_shape = grad_input_desc->shape();
        info.weight_grad_shape = grad_weight_desc->shape();
        if (has_bias) {
            info.bias_grad_shape = grad_bias_desc->shape();
        }
        info.output_grad_shape = grad_output_desc->shape();
        info.input_shape = input_desc->shape();
        info.weight_shape = weight_desc->shape();
        info.input_std_deviation_shape = input_std_deviation_desc->shape();
        info.input_standardization_shape = input_standardization_desc->shape();
        info.shape = grad_output_desc->shape();
        
        info.input_grad_strides = grad_input_desc->strides();
        info.weight_grad_strides = grad_weight_desc->strides();
        if (has_bias) {
            info.bias_grad_strides = grad_bias_desc->strides();
        }
        info.output_grad_strides = grad_output_desc->strides();
        info.input_strides = input_desc->strides();
        info.weight_strides = weight_desc->strides();
        info.input_std_deviation_strides = input_std_deviation_desc->strides();
        info.input_standardization_strides = input_standardization_desc->strides();
        
        return utils::Result<LayerNormBackwardInfo>(std::move(info));
    }
};

} // namespace op::layer_norm_backward

#endif // __LAYER_NORM_BACKWARD_H__