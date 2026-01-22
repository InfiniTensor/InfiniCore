#include "infinicore/ops/topk.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<TopK::schema> &TopK::dispatcher() {
    static common::OpDispatcher<TopK::schema> dispatcher_;
    return dispatcher_;
};

void TopK::execute(Tensor input, Tensor values, Tensor indices, int k, int dim, bool largest, bool sorted) {
    infinicore::context::setDevice(input->device());
    dispatcher().lookup(input->device().getType())(input, values, indices, k, dim, largest, sorted);
}

std::tuple<Tensor, Tensor> topk(Tensor input, int k, int dim, bool largest, bool sorted) {

    int ndim = input->ndim();
    int normalized_dim = dim;
    if (normalized_dim < 0) {
        normalized_dim = ndim + normalized_dim;
    }

    Shape output_shape = input->shape();

    if (k > output_shape[normalized_dim]) {
        throw std::runtime_error("k cannot be larger than the size of the dimension.");
    }
    output_shape[normalized_dim] = k;

    auto values = Tensor::empty(output_shape, input->dtype(), input->device());
    auto indices = Tensor::empty(output_shape, DataType::I64, input->device());

    topk_(input, values, indices, k, normalized_dim, largest, sorted);
    return {values, indices};
}

void topk_(Tensor input, Tensor values, Tensor indices, int k, int dim, bool largest, bool sorted) {
    int ndim = input->ndim();
    if (dim < 0) {
        dim = ndim + dim;
    }
    TopK::execute(input, values, indices, k, dim, largest, sorted);
}

} // namespace infinicore::op