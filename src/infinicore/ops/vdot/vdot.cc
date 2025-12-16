#include "infinicore/ops/vdot.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<Vdot::schema> &Vdot::dispatcher() {
    static common::OpDispatcher<Vdot::schema> dispatcher_;
    return dispatcher_;
}

void Vdot::execute(Tensor out, Tensor a, Tensor b) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, a, b);

    // Inputs must be 1D and same length
    if (a->ndim() != 1 || b->ndim() != 1) {
        throw std::runtime_error("vdot: input tensors must be 1D");
    }
    if (a->shape()[0] != b->shape()[0]) {
        throw std::runtime_error("vdot: input tensors must have the same length");
    }

    infinicore::context::setDevice(out->device());
    auto device_type = out->device().getType();
    auto func = dispatcher().lookup(device_type);

    if (func == nullptr) {
        throw std::runtime_error(
            "No Vdot implementation found for device type: "
            + std::to_string(static_cast<int>(device_type)));
    }

    func(out, a, b);
}

Tensor vdot(Tensor a, Tensor b) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(a, b);

    // Output dtype equals input dtype for now
    auto out = Tensor::empty({}, a->dtype(), a->device());
    vdot_(out, a, b);
    return out;
}

void vdot_(Tensor out, Tensor a, Tensor b) {
    Vdot::execute(out, a, b);
}

} // namespace infinicore::op
