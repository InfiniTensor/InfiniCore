#include "infinicore/ops/nll_loss.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<NLLLoss::schema> &NLLLoss::dispatcher() {
    static common::OpDispatcher<NLLLoss::schema> dispatcher_;
    return dispatcher_;
};

void NLLLoss::execute(Tensor input, Tensor target, std::optional<Tensor> weight, Tensor output, int64_t ignore_index) {
    infinicore::context::setDevice(input->device());
    dispatcher().lookup(input->device().getType())(input, target, weight, output, ignore_index);
}

Tensor nll_loss(Tensor input, Tensor target, std::optional<Tensor> weight, int64_t ignore_index) {
    auto output = Tensor::empty({}, input->dtype(), input->device());
    nll_loss_(input, target, weight, output, ignore_index);
    return output;
}

void nll_loss_(Tensor input, Tensor target, std::optional<Tensor> weight, Tensor output, int64_t ignore_index) {
    NLLLoss::execute(input, target, weight, output, ignore_index);
}

} // namespace infinicore::op