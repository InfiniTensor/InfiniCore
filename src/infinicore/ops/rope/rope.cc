#include "infinicore/ops/rope.hpp"
#include "infinicore/context/context.hpp"
#include <stdexcept>

namespace infinicore::op {

common::OpDispatcher<RoPE::schema> &RoPE::dispatcher() {
    static common::OpDispatcher<RoPE::schema> dispatcher_;
    return dispatcher_;
};

void RoPE::execute(Tensor x_out, const Tensor &x, const Tensor &pos, const Tensor &sin_cache, const Tensor &cos_cache, infiniopRoPEAlgo_t algo) {
    auto device_type = context::getDevice().getType();
    auto func = dispatcher().lookup(device_type);

    if (func == nullptr) {
        throw std::runtime_error("No RoPE implementation found for device type: " + std::to_string(static_cast<int>(device_type)));
    }

    func(x_out, x, pos, sin_cache, cos_cache, algo);
}

void rope_(Tensor x_out, const Tensor &x, const Tensor &pos, const Tensor &sin_cache, const Tensor &cos_cache, infiniopRoPEAlgo_t algo) {
    RoPE::execute(x_out, x, pos, sin_cache, cos_cache, algo);
}

Tensor rope(const Tensor &x, const Tensor &pos, const Tensor &sin_cache, const Tensor &cos_cache, infiniopRoPEAlgo_t algo) {
    Shape shape = x->shape();
    auto x_out = Tensor::empty(shape, x->dtype(), x->device());
    rope_(x_out, x, pos, sin_cache, cos_cache, algo);
    return x_out;
}

} // namespace infinicore::op
