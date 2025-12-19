#include "infinicore/ops/index_select.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<IndexSelect::schema> &IndexSelect::dispatcher() {
    static common::OpDispatcher<IndexSelect::schema> dispatcher_;
    return dispatcher_;
};

void IndexSelect::execute(Tensor output, Tensor input, int dim, Tensor index) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input, index);
    infinicore::context::setDevice(output->device(), true);
    dispatcher().lookup(output->device().getType())(output, input, dim, index);
}

Tensor index_select(Tensor input, int dim, Tensor index) {
    if (index->ndim() != 1) {
        throw std::runtime_error("Index tensor must be 1-dimensional for index_select operation.");
    }

    if (dim < 0) {
        dim = dim + input->ndim();
    }

    if (dim < 0 || dim >= static_cast<int>(input->ndim())) {
        throw std::runtime_error("Dimension out of range for index_select operation.");
    }

    auto output_shape = input->shape();
    output_shape[dim] = index->shape()[0];

    auto output = Tensor::empty(output_shape, input->dtype(), input->device());
    index_select_(output, input, dim, index);
    return output;
}

void index_select_(Tensor output, Tensor input, int dim, Tensor index) {
    IndexSelect::execute(output, input, dim, index);
}

} // namespace infinicore::op
