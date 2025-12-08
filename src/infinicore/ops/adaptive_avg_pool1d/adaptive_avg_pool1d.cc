#include "infinicore/ops/adaptive_avg_pool1d.hpp"
#include <stdexcept>
#include <vector>

namespace infinicore::op {

// =========================================================
// Dispatcher & Execute
// =========================================================

common::OpDispatcher<AdaptiveAvgPool1d::schema> &AdaptiveAvgPool1d::dispatcher() {
    static common::OpDispatcher<AdaptiveAvgPool1d::schema> dispatcher_;
    return dispatcher_;
};

void AdaptiveAvgPool1d::execute(Tensor output, Tensor input) {
    auto device_type = context::getDevice().getType();
    auto func = dispatcher().lookup(device_type);

    if (func == nullptr) {
        throw std::runtime_error("No AdaptiveAvgPool1d implementation found for device type: " + std::to_string(static_cast<int>(device_type)));
    }

    func(output, input);
}

// =========================================================
// Wrapper Implementation
// =========================================================

Tensor adaptive_avg_pool1d(Tensor input, int64_t output_size) {
    // 1. 维度检查
    size_t ndim = input->ndim();
    if (ndim != 2 && ndim != 3) {
        throw std::runtime_error("AdaptiveAvgPool1d: Input tensor must be 2D or 3D.");
    }

    // 2. 参数检查
    if (output_size <= 0) {
        throw std::runtime_error("AdaptiveAvgPool1d: output_size must be positive.");
    }

    // 3. 推导输出形状
    // 复制输入形状 (Shape 本质通常是 std::vector<size_t>)
    auto out_shape = input->shape();
    // 修改最后一个维度为 output_size
    out_shape[ndim - 1] = output_size;

    // 4. 创建输出 Tensor
    // 直接使用 input->dtype()，不需要引入 DType 定义进行额外检查
    auto output = Tensor::empty(out_shape, input->dtype(), input->device());

    // 5. 执行计算
    // execute 内部会根据 input 和 output 的形状自动进行窗口计算
    AdaptiveAvgPool1d::execute(output, input);

    return output;
}

} // namespace infinicore::op