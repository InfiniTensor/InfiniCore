#include "infinicore/ops/bilinear.hpp"

namespace infinicore::op {

common::OpDispatcher<Bilinear::schema> &Bilinear::dispatcher() {
    static common::OpDispatcher<Bilinear::schema> dispatcher_;
    return dispatcher_;
};

void Bilinear::execute(Tensor out, Tensor x1, Tensor x2, Tensor weight, Tensor bias) {
    dispatcher().lookup(context::getDevice().getType())(out, x1, x2, weight, bias);
}

Tensor bilinear(Tensor x1, Tensor x2, Tensor weight, Tensor bias) {
    size_t batch_size = x1->shape()[0];
    size_t out_features = weight->shape()[0];
    Shape shape = {batch_size, out_features};
    auto out = Tensor::empty(shape, x1->dtype(), x1->device());
    bilinear_(out, x1, x2, weight, bias);
    return out;
}

void bilinear_(Tensor out, Tensor x1, Tensor x2, Tensor weight, Tensor bias) {
    Bilinear::execute(out, x1, x2, weight, bias);
}

} // namespace infinicore::op