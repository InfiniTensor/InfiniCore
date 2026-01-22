#include "infinicore/ops/sum.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<SumGlobal::schema> &SumGlobal::dispatcher() {
    static common::OpDispatcher<SumGlobal::schema> dispatcher_;
    return dispatcher_;
};

void SumGlobal::execute(Tensor input, Tensor output) {
    infinicore::context::setDevice(input->device());
    dispatcher().lookup(input->device().getType())(input, output);
}

Tensor sum_global(Tensor input) {
    Shape shape = Shape();
    auto output = Tensor::empty(shape, input->dtype(), input->device());
    sum_global_(input, output);
    return output;
}

void sum_global_(Tensor input, Tensor output) {
    SumGlobal::execute(input, output);
}

common::OpDispatcher<SumReduce::schema> &SumReduce::dispatcher() {
    static common::OpDispatcher<SumReduce::schema> dispatcher_;
    return dispatcher_;
};

void SumReduce::execute(Tensor input, Tensor output, int dim, bool keepdim) {
    infinicore::context::setDevice(input->device());
    dispatcher().lookup(input->device().getType())(input, output, dim, keepdim);
}

Tensor sum_reduce(Tensor input, int dim, bool keepdim) {
    int normalized_dim = dim;
    if (normalized_dim < 0) {
        normalized_dim = input->ndim() + normalized_dim;
    }

    Shape output_shape;
    const auto &input_shape = input->shape();

    if (keepdim) {
        output_shape = input_shape;
        output_shape[normalized_dim] = 1;
    } else {
        for (int i = 0; i < static_cast<int>(input_shape.size()); ++i) {
            if (i != normalized_dim) {
                output_shape.push_back(input_shape[i]);
            }
        }
    }

    auto output = Tensor::empty(output_shape, input->dtype(), input->device());
    sum_reduce_(input, output, dim, keepdim);
    return output;
}

void sum_reduce_(Tensor input, Tensor output, int dim, bool keepdim) {
    SumReduce::execute(input, output, dim, keepdim);
}

} // namespace infinicore::op