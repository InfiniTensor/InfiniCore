#ifndef __LAYER_NORM_H__
#define __LAYER_NORM_H__

#include "../../operator.h"
#include "../../tensor.h"
#include "infinicore.h"

namespace op::layer_norm {

struct LayerNormInfo {
    size_t _batch_size;         // 批次大小
    size_t _normalized_size;    // 归一化维度大小最后一维的大小，用于归一化
    size_t total_elements;      // 总元素数量
    size_t input_size;
    size_t output_size;
    
    infiniDtype_t dtype;
    infiniDtype_t wtype;        // 权重数据类型
    infiniDtype_t btype;        // 偏置数据类型
    infiniDtype_t atype;        // 激活数据类型
    float eps;                  // epsilon值
    float epsilon;              // epsilon值（兼容参考实现）
    bool has_bias;              // 是否有bias参数
    
    std::vector<size_t> input_shape;
    std::vector<size_t> output_shape;
    std::vector<size_t> weight_shape;
    std::vector<size_t> bias_shape;
    std::vector<size_t> input_std_deviation_shape;
    std::vector<size_t> input_standardization_shape;
    std::vector<size_t> shape;  // 输出形状（兼容参考实现）
    
    std::vector<ptrdiff_t> input_strides;
    std::vector<ptrdiff_t> output_strides;
    std::vector<ptrdiff_t> weight_strides;
    std::vector<ptrdiff_t> bias_strides;
    std::vector<ptrdiff_t> input_std_deviation_strides;
    std::vector<ptrdiff_t> input_standardization_strides;
    
    // 兼容参考实现的方法
    size_t ndim() const { return shape.size(); }
    size_t dim() const { return _normalized_size; }
    size_t batch_size() const {
        return _batch_size;
    }
};

} // namespace op::layer_norm

#endif // __LAYER_NORM_H__