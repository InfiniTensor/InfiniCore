#include "infinicore/ops/where.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<Where::schema> &Where::dispatcher() {
    static common::OpDispatcher<Where::schema> dispatcher_;
    return dispatcher_;
}

void Where::execute(Tensor out, Tensor cond, Tensor x, Tensor y) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, cond, x, y);
    infinicore::context::setDevice(out->device());
    auto device_type = out->device().getType();
    auto func = dispatcher().lookup(device_type);

    if (func == nullptr) {
        throw std::runtime_error(
            "No Where implementation found for device type: "
            + std::to_string(static_cast<int>(device_type)));
    }

    func(out, cond, x, y);
}

Tensor where(Tensor cond, Tensor x, Tensor y) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(cond, x, y);
    // Output dtype follows x/y dtype
    auto out = Tensor::empty(x->shape(), x->dtype(), x->device());
    where_(out, cond, x, y);
    return out;
}

void where_(Tensor out, Tensor cond, Tensor x, Tensor y) {
    Where::execute(out, cond, x, y);
}

} // namespace infinicore::op


