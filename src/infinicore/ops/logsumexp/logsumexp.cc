#include "infinicore/ops/logsumexp.hpp"
#include <iostream>
#include <stdexcept>

namespace infinicore::op {

common::OpDispatcher<LogSumExp::schema> &LogSumExp::dispatcher() {
    static common::OpDispatcher<LogSumExp::schema> dispatcher_;
    return dispatcher_;
};

void LogSumExp::execute(Tensor input, int dim, bool keepdim, Tensor output) {
    infinicore::context::setDevice(input->device(), true);
    auto device_type = context::getDevice().getType();
    auto func = dispatcher().lookup(device_type);

    if (func == nullptr) {
        throw std::runtime_error("No LogSumExp implementation found for device type: " + std::to_string(static_cast<int>(device_type)));
    }

    func(input, dim, keepdim, output);
}

Tensor logsumexp(Tensor input, int dim, bool keepdim) {
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
    logsumexp_(input, dim, keepdim, output);
    return output;
}

void logsumexp_(Tensor input, int dim, bool keepdim, Tensor output) {
    LogSumExp::execute(input, dim, keepdim, output);
}
} // namespace infinicore::op
