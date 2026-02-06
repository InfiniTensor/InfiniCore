#include "infinicore/ops/addcdiv.hpp"
#include "../../utils.hpp"
#include <algorithm>

namespace infinicore::op {

common::OpDispatcher<Addcdiv::schema> &Addcdiv::dispatcher() {
    static common::OpDispatcher<Addcdiv::schema> dispatcher_;
    return dispatcher_;
};

void Addcdiv::execute(Tensor input, Tensor t1, Tensor t2, Tensor output, float value) {
    infinicore::context::setDevice(input->device());
    dispatcher().lookup(input->device().getType())(input, t1, t2, output, value);
}

static Shape broadcast_shape_3(const Shape &a, const Shape &b, const Shape &c) {
    int ndim = std::max({a.size(), b.size(), c.size()});
    Shape out_shape;

    for (int i = 0; i < ndim; ++i) {
        int dim_a = (i < ndim - a.size()) ? 1 : a[i - (ndim - a.size())];
        int dim_b = (i < ndim - b.size()) ? 1 : b[i - (ndim - b.size())];
        int dim_c = (i < ndim - c.size()) ? 1 : c[i - (ndim - c.size())];

        int target = std::max({dim_a, dim_b, dim_c});

        if ((dim_a != target && dim_a != 1) || (dim_b != target && dim_b != 1) || (dim_c != target && dim_c != 1)) {
            throw std::runtime_error("Shapes are not broadcastable");
        }
        out_shape.push_back(target);
    }
    return out_shape;
}

Tensor addcdiv(Tensor input, Tensor t1, Tensor t2, float value) {
    Shape out_shape = broadcast_shape_3(input->shape(), t1->shape(), t2->shape());
    auto output = Tensor::empty(out_shape, input->dtype(), input->device());
    addcdiv_(input, t1, t2, output, value);
    return output;
}

void addcdiv_(Tensor input, Tensor t1, Tensor t2, Tensor output, float value) {
    Addcdiv::execute(input, t1, t2, output, value);
}

} // namespace infinicore::op