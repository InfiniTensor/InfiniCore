#include "infinicore/ops/rope.hpp"

#include "../../utils.hpp"
#include "infinicore/context/context.hpp"

#include <stdexcept>

namespace infinicore::op {

common::OpDispatcher<RoPE::schema> &RoPE::dispatcher() {
    static common::OpDispatcher<RoPE::schema> dispatcher_;
    return dispatcher_;
};

void RoPE::execute(Tensor x_out, const Tensor &x, const Tensor &pos, const Tensor &sin_table, const Tensor &cos_table, infinicore::nn::RoPE::Algo algo) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(x_out, x, pos, sin_table, cos_table);
    infinicore::context::setDevice(x_out->device());
    auto device_type = x_out->device().getType();
    auto func = dispatcher().lookup(device_type);

    if (func == nullptr) {
        throw std::runtime_error("No RoPE implementation found for device type: " + std::to_string(static_cast<int>(device_type)));
    }

    func(x_out, x, pos, sin_table, cos_table, algo);
}

void rope_(Tensor x_out, const Tensor &x, const Tensor &pos, const Tensor &sin_table, const Tensor &cos_table, infinicore::nn::RoPE::Algo algo) {
    RoPE::execute(x_out, x, pos, sin_table, cos_table, algo);
}

Tensor rope(const Tensor &x, const Tensor &pos, const Tensor &sin_table, const Tensor &cos_table, infinicore::nn::RoPE::Algo algo) {
    Shape shape = x->shape();
    auto x_out = Tensor::empty(shape, x->dtype(), x->device());
    rope_(x_out, x, pos, sin_table, cos_table, algo);
    return x_out;
}

} // namespace infinicore::op
