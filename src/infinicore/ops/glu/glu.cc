#include "infinicore/ops/glu.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<Glu::schema> &Glu::dispatcher() {
    static common::OpDispatcher<Glu::schema> dispatcher_;
    return dispatcher_;
}

void Glu::execute(Tensor input, Tensor output, int dim) {
    infinicore::context::setDevice(input->device());
    dispatcher().lookup(input->device().getType())(input, output, dim);
}

Tensor glu(Tensor input, int dim) {
    if (dim < 0) {
        dim += input->ndim();
    }
    auto out_shape = input->shape();
    out_shape[dim] /= 2;
    auto output = Tensor::empty(out_shape, input->dtype(), input->device());
    glu_(input, output, dim);
    return output;
}

void glu_(Tensor input, Tensor output, int dim) {
    Glu::execute(input, output, dim);
}

} // namespace infinicore::op