#include "infinicore/ops/select_scatter.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<SelectScatter::schema> &SelectScatter::dispatcher() {
    static common::OpDispatcher<SelectScatter::schema> dispatcher_;
    return dispatcher_;
};

void SelectScatter::execute(Tensor input, Tensor src, int64_t dim, int64_t index, Tensor output) {
    infinicore::context::setDevice(input->device());
    dispatcher().lookup(input->device().getType())(input, src, dim, index, output);
}

Tensor select_scatter(Tensor input, Tensor src, int64_t dim, int64_t index) {
    auto output = Tensor::empty(input->shape(), input->dtype(), input->device());
    select_scatter_(input, src, dim, index, output);
    return output;
}

void select_scatter_(Tensor input, Tensor src, int64_t dim, int64_t index, Tensor output) {
    SelectScatter::execute(input, src, dim, index, output);
}

} // namespace infinicore::op