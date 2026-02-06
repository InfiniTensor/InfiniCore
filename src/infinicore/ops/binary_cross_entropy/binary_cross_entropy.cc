#include "infinicore/ops/binary_cross_entropy.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<BinaryCrossEntropy::schema> &BinaryCrossEntropy::dispatcher() {
    static common::OpDispatcher<BinaryCrossEntropy::schema> dispatcher_;
    return dispatcher_;
};

void BinaryCrossEntropy::execute(Tensor input, Tensor target, std::optional<Tensor> weight, Tensor output, std::string reduction) {
    infinicore::context::setDevice(input->device());
    dispatcher().lookup(input->device().getType())(input, target, weight, output, reduction);
}

Tensor binary_cross_entropy(Tensor input, Tensor target, std::optional<Tensor> weight, std::string reduction) {
    Shape output_shape = {};
    if (reduction == "none") {
        output_shape = input->shape();
    }

    auto output = Tensor::empty(output_shape, input->dtype(), input->device());
    binary_cross_entropy_(input, target, weight, output, reduction);
    return output;
}

void binary_cross_entropy_(Tensor input, Tensor target, std::optional<Tensor> weight, Tensor output, std::string reduction) {
    BinaryCrossEntropy::execute(input, target, weight, output, reduction);
}

} // namespace infinicore::op