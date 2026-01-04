#include "infinicore/ops/cross_entropy.hpp"

#include "../../utils.hpp"

#include <stdexcept>

namespace infinicore::op {

// 1. 实现分发器单例
common::OpDispatcher<CrossEntropy::schema> &CrossEntropy::dispatcher() {
    static common::OpDispatcher<CrossEntropy::schema> dispatcher_;
    return dispatcher_;
};

// 2. 实现统一执行入口
void CrossEntropy::execute(Tensor output, Tensor input, Tensor target) {
    // 检查所有 Tensor 是否在同一设备
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input);
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(input, target);

    infinicore::context::setDevice(output->device());
    auto device_type = output->device().getType();
    
    // 查找对应后端的实现
    auto func = dispatcher().lookup(device_type);

    if (func == nullptr) {
        throw std::runtime_error("No CrossEntropy implementation found for device type: " + std::to_string(static_cast<int>(device_type)));
    }

    // 执行计算
    func(output, input, target);
}

// 3. 实现非原地接口 (自动创建 Output)
Tensor cross_entropy(Tensor input, Tensor target) {
    // 逻辑：CrossEntropy 的输出形状通常与 Target 形状一致
    // Input: [Batch, Seq, Vocab]
    // Target: [Batch, Seq]
    // Output: [Batch, Seq] (Per-token Loss)
    Shape shape = target->shape();
    
    // Output 的数据类型通常跟随 Input (Logits) 的浮点类型 (F16/F32)，而不是 Target 的整型
    auto output = Tensor::empty(shape, input->dtype(), input->device());
    
    cross_entropy_(output, input, target);
    return output;
}

// 4. 实现显式输出接口
void cross_entropy_(Tensor output, Tensor input, Tensor target) {
    CrossEntropy::execute(output, input, target);
}

} // namespace infinicore::op