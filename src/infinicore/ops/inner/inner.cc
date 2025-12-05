#include <infinicore/ops/inner.hpp>
#include <stdexcept>

namespace infinicore::op {


common::OpDispatcher<Inner::schema> &Inner::dispatcher() {
    static common::OpDispatcher<Inner::schema> dispatcher_;
    return dispatcher_;
};

void Inner::execute(Tensor input, Tensor other, Tensor out) {
    auto device_type = context::getDevice().getType();
    auto func = dispatcher().lookup(device_type);

    if (func == nullptr) {
        throw std::runtime_error("No Inner implementation found for device type: " + std::to_string(static_cast<int>(device_type)));
    }

    func(input, other, out);
}

Tensor inner(Tensor input, Tensor other) {
    size_t input_ndim = input->ndim();
    size_t other_ndim = other->ndim();

    assert(input->shape()[input_ndim - 1] == other->shape()[other_ndim - 1]);

    Shape out_shape;
    for (int i = 0; i < input_ndim - 1; i ++)
        out_shape.push_back(input->shape()[i]);
    for (int i = 0; i < other_ndim - 1; i ++)
        out_shape.push_back(other->shape()[i]);
    auto out = Tensor::zeros(out_shape, input->dtype(), input->device());

    inner_(input, other, out);
    return out;
}

void inner_(Tensor input, Tensor other, Tensor out) {
    
    Inner::execute(input, other, out);

}

}