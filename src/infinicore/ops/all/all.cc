#include "infinicore/ops/all.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<AllGlobal::schema> &AllGlobal::dispatcher() {
    static common::OpDispatcher<AllGlobal::schema> dispatcher_;
    return dispatcher_;
};

void AllGlobal::execute(Tensor input, Tensor output) {
    infinicore::context::setDevice(input->device());
    dispatcher().lookup(input->device().getType())(input, output);
}

Tensor all_global(Tensor input) {
    Shape shape = Shape();
    auto output = Tensor::empty(shape, DataType::BOOL, input->device());
    all_global_(input, output);
    return output;
}

void all_global_(Tensor input, Tensor output) {
    AllGlobal::execute(input, output);
}

common::OpDispatcher<AllReduce::schema> &AllReduce::dispatcher() {
    static common::OpDispatcher<AllReduce::schema> dispatcher_;
    return dispatcher_;
};

void AllReduce::execute(Tensor input, Tensor output, int dim, bool keepdim) {
    infinicore::context::setDevice(input->device());
    dispatcher().lookup(input->device().getType())(input, output, dim, keepdim);
}

Tensor all_reduce(Tensor input, int dim, bool keepdim) {
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

    auto output = Tensor::empty(output_shape, DataType::BOOL, input->device());
    all_reduce_(input, output, dim, keepdim);
    return output;
}

void all_reduce_(Tensor input, Tensor output, int dim, bool keepdim) {
    AllReduce::execute(input, output, dim, keepdim);
}

} // namespace infinicore::op