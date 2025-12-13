#include "infinicore/ops/max_reduce.hpp"

#include "../../utils.hpp"
#include "infinicore/dtype.hpp"

namespace infinicore::op {

common::OpDispatcher<MaxReduce::schema> &MaxReduce::dispatcher() {
    static common::OpDispatcher<MaxReduce::schema> dispatcher_;
    return dispatcher_;
};

void MaxReduce::execute(Tensor input, Tensor output, Tensor indices, int dim, bool keepdim) {
    infinicore::context::setDevice(input->device(), true);
    dispatcher().lookup(input->device().getType())(input, output, indices, dim, keepdim);
}

std::tuple<Tensor, Tensor> max_reduce(Tensor input, int dim, bool keepdim) {
    // 规范化 dim
    int normalized_dim = dim;
    if (normalized_dim < 0) {
        normalized_dim = input->ndim() + normalized_dim;
    }

    // 计算输出形状
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
    auto indices = Tensor::empty(output_shape, DataType::I64, input->device());
    max_reduce_(input, output, indices, dim, keepdim);
    return {output, indices};
}

void max_reduce_(Tensor input, Tensor output, Tensor indices, int dim, bool keepdim) {
    MaxReduce::execute(input, output, indices, dim, keepdim);
}

} // namespace infinicore::op
