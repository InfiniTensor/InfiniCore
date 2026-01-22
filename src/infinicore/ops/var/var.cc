#include "infinicore/ops/var.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<VarGlobal::schema> &VarGlobal::dispatcher() {
    static common::OpDispatcher<VarGlobal::schema> dispatcher_;
    return dispatcher_;
};

void VarGlobal::execute(Tensor input, Tensor output, int correction) {
    infinicore::context::setDevice(input->device());
    dispatcher().lookup(input->device().getType())(input, output, correction);
}

Tensor var_global(Tensor input, int correction) {
    Shape shape = Shape();
    auto output = Tensor::empty(shape, input->dtype(), input->device());
    var_global_(input, output, correction);
    return output;
}

void var_global_(Tensor input, Tensor output, int correction) {
    VarGlobal::execute(input, output, correction);
}

common::OpDispatcher<VarReduce::schema> &VarReduce::dispatcher() {
    static common::OpDispatcher<VarReduce::schema> dispatcher_;
    return dispatcher_;
};

void VarReduce::execute(Tensor input, Tensor output, int dim, int correction, bool keepdim) {
    infinicore::context::setDevice(input->device());
    dispatcher().lookup(input->device().getType())(input, output, dim, correction, keepdim);
}

Tensor var_reduce(Tensor input, int dim, int correction, bool keepdim) {
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
    var_reduce_(input, output, dim, correction, keepdim);
    return output;
}

void var_reduce_(Tensor input, Tensor output, int dim, int correction, bool keepdim) {
    VarReduce::execute(input, output, dim, correction, keepdim);
}

} // namespace infinicore::op