#include "infinicore/ops/diagflat.hpp"

namespace infinicore::op {

common::OpDispatcher<Diagflat::schema> &Diagflat::dispatcher() {
    static common::OpDispatcher<Diagflat::schema> dispatcher_;
    return dispatcher_;
}

void Diagflat::execute(Tensor output, Tensor input, int64_t offset) {
    dispatcher().lookup(context::getDevice().getType())(output, input, offset);
}

Tensor diagflat(Tensor input, int64_t offset) {
    // 输出 shape 由后端决定，这里直接让后端写入 output
    // 先构造一个占位 Tensor（0-dim），再让实现自己 resize/allocate 也可以；
    // 为简单起见，这里暂时只支持 out 版本：用户通过 diagflat_ 使用。
    auto flat = Tensor::empty({input->numel()}, input->dtype(), input->device());
    diagflat_(flat, input, offset);
    return flat;
}

void diagflat_(Tensor output, Tensor input, int64_t offset) {
    Diagflat::execute(output, input, offset);
}

} // namespace infinicore::op


