#include "infinicore/ops/addbmm.hpp"
#include "infinicore/ops/addbmm.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

// 1. 初始化 Dispatcher
common::OpDispatcher<Addbmm::schema> &Addbmm::dispatcher() {
    static common::OpDispatcher<Addbmm::schema> dispatcher_;
    return dispatcher_;
};

// 2. 类层面的 execute 方法
// 【关键修复】参数顺序改为 float beta, float alpha
void Addbmm::execute(Tensor output, Tensor input, Tensor batch1, Tensor batch2, float beta, float alpha) {
    
    // 切换上下文
    infinicore::context::setDevice(output->device());

    // 分发计算
    dispatcher().lookup(output->device().getType())(output, input, batch1, batch2, beta, alpha);
}

// 3. Out-of-place 接口
// 【关键修复】参数顺序改为 float beta, float alpha
Tensor addbmm(Tensor input, Tensor batch1, Tensor batch2, float beta, float alpha) {
    auto output = Tensor::empty(input->shape(), input->dtype(), input->device());
    Addbmm::execute(output, input, batch1, batch2, beta, alpha);
    return output;
}

// 4. In-place 接口
// 【关键修复】参数顺序改为 float beta, float alpha
void addbmm_(Tensor output, Tensor input, Tensor batch1, Tensor batch2, float beta, float alpha) {
    Addbmm::execute(output, input, batch1, batch2, beta, alpha);
}

} // namespace infinicore::op