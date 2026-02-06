#include "infinicore/ops/minimum.hpp"
#include "../../utils.hpp"
#include <algorithm>

namespace infinicore::op {

common::OpDispatcher<Minimum::schema> &Minimum::dispatcher() {
    static common::OpDispatcher<Minimum::schema> dispatcher_;
    return dispatcher_;
};

void Minimum::execute(Tensor input, Tensor other, Tensor output) {
    infinicore::context::setDevice(input->device());
    dispatcher().lookup(input->device().getType())(input, other, output);
}

static Shape broadcast_shape(const Shape &a, const Shape &b) {
    Shape out_shape;
    int ndim_a = a.size();
    int ndim_b = b.size();
    int max_ndim = std::max(ndim_a, ndim_b);

    for (int i = 0; i < max_ndim; ++i) {
        int dim_a = (i < max_ndim - ndim_a) ? 1 : a[i - (max_ndim - ndim_a)];
        int dim_b = (i < max_ndim - ndim_b) ? 1 : b[i - (max_ndim - ndim_b)];

        if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
            throw std::runtime_error("Shapes are not broadcastable");
        }
        out_shape.push_back(std::max(dim_a, dim_b));
    }
    return out_shape;
}

Tensor minimum(Tensor input, Tensor other) {
    Shape out_shape = broadcast_shape(input->shape(), other->shape());
    auto output = Tensor::empty(out_shape, input->dtype(), input->device());
    minimum_(input, other, output);
    return output;
}

void minimum_(Tensor input, Tensor other, Tensor output) {
    Minimum::execute(input, other, output);
}

} // namespace infinicore::op