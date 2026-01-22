#include "infinicore/ops/var_mean.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<VarMeanGlobal::schema> &VarMeanGlobal::dispatcher() {
    static common::OpDispatcher<VarMeanGlobal::schema> dispatcher_;
    return dispatcher_;
};

void VarMeanGlobal::execute(Tensor input, Tensor out_var, Tensor out_mean, int correction) {
    infinicore::context::setDevice(input->device());
    dispatcher().lookup(input->device().getType())(input, out_var, out_mean, correction);
}

std::tuple<Tensor, Tensor> var_mean_global(Tensor input, int correction) {
    Shape shape = Shape();
    auto out_var = Tensor::empty(shape, input->dtype(), input->device());
    auto out_mean = Tensor::empty(shape, input->dtype(), input->device());
    var_mean_global_(input, out_var, out_mean, correction);
    return {out_var, out_mean};
}

void var_mean_global_(Tensor input, Tensor out_var, Tensor out_mean, int correction) {
    VarMeanGlobal::execute(input, out_var, out_mean, correction);
}

common::OpDispatcher<VarMeanReduce::schema> &VarMeanReduce::dispatcher() {
    static common::OpDispatcher<VarMeanReduce::schema> dispatcher_;
    return dispatcher_;
};

void VarMeanReduce::execute(Tensor input, Tensor out_var, Tensor out_mean, int dim, int correction, bool keepdim) {
    infinicore::context::setDevice(input->device());
    dispatcher().lookup(input->device().getType())(input, out_var, out_mean, dim, correction, keepdim);
}

std::tuple<Tensor, Tensor> var_mean_reduce(Tensor input, int dim, int correction, bool keepdim) {
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

    auto out_var = Tensor::empty(output_shape, input->dtype(), input->device());
    auto out_mean = Tensor::empty(output_shape, input->dtype(), input->device());

    var_mean_reduce_(input, out_var, out_mean, dim, correction, keepdim);
    return {out_var, out_mean};
}

void var_mean_reduce_(Tensor input, Tensor out_var, Tensor out_mean, int dim, int correction, bool keepdim) {
    VarMeanReduce::execute(input, out_var, out_mean, dim, correction, keepdim);
}

} // namespace infinicore::op