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
    // 根据 PyTorch 语义：先展平，长度 n = input.numel()
    // 输出为 2D 矩阵 (n + abs(offset), n + abs(offset))
    auto n = input->numel();
    auto abs_off = offset >= 0 ? static_cast<size_t>(offset)
                               : static_cast<size_t>(-offset);
    auto dim = static_cast<Size>(n + abs_off);

    auto out = Tensor::empty({dim, dim}, input->dtype(), input->device());
    diagflat_(out, input, offset);
    return out;
}

void diagflat_(Tensor output, Tensor input, int64_t offset) {
    Diagflat::execute(output, input, offset);
}

} // namespace infinicore::op
