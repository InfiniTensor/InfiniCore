#include "infinicore/ops/bucketize.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<Bucketize::schema> &Bucketize::dispatcher() {
    static common::OpDispatcher<Bucketize::schema> dispatcher_;
    return dispatcher_;
};

void Bucketize::execute(Tensor input, Tensor boundaries, Tensor output, bool right) {
    infinicore::context::setDevice(input->device());
    dispatcher().lookup(input->device().getType())(input, boundaries, output, right);
}

Tensor bucketize(Tensor input, Tensor boundaries, bool right) {
    auto output = Tensor::empty(input->shape(), DataType::I64, input->device());
    bucketize_(input, boundaries, output, right);
    return output;
}

void bucketize_(Tensor input, Tensor boundaries, Tensor output, bool right) {
    Bucketize::execute(input, boundaries, output, right);
}

} // namespace infinicore::op