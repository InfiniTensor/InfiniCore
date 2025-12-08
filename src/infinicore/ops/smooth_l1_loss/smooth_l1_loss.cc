#include "infinicore/ops/smooth_l1_loss.hpp"

namespace infinicore::op {

// 1. 定义 Dispatcher 单例
common::OpDispatcher<SmoothL1Loss::schema> &SmoothL1Loss::dispatcher() {
    static common::OpDispatcher<SmoothL1Loss::schema> dispatcher_;
    return dispatcher_;
};

// 2. Execute 实现：查找对应设备的核函数并执行
void SmoothL1Loss::execute(Tensor output, Tensor input, Tensor target, float beta, int64_t reduction) {
    dispatcher().lookup(context::getDevice().getType())(output, input, target, beta, reduction);
}

// 3. 函数式接口
Tensor smooth_l1_loss(Tensor input, Tensor target, float beta, int64_t reduction) {
    // 【关键逻辑】根据 reduction 决定输出形状
    Shape output_shape;
    if (reduction == 0) { 
        // Reduction::None -> 输出形状与输入一致
        output_shape = input->shape();
    } else { 
        output_shape = {}; 
    }

    auto output = Tensor::empty(output_shape, input->dtype(), input->device());
    
    smooth_l1_loss_(output, input, target, beta, reduction);
    return output;
}

void smooth_l1_loss_(Tensor output, Tensor input, Tensor target, float beta, int64_t reduction) {
    SmoothL1Loss::execute(output, input, target, beta, reduction);
}

} // namespace infinicore::op