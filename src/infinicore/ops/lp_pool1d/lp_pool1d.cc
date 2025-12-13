#include "infinicore/ops/lp_pool1d.hpp"
#include <cmath>
#include <stdexcept>

namespace infinicore::op {

common::OpDispatcher<Lp_Pool1d::schema> &Lp_Pool1d::dispatcher() {
    static common::OpDispatcher<Lp_Pool1d::schema> dispatcher_;
    return dispatcher_;
};

void Lp_Pool1d::execute(Tensor output, Tensor input, float norm_type, size_t kernel_size, size_t stride, bool ceil_mode) {
    infinicore::context::setDevice(input->device(), true);
    auto device_type = context::getDevice().getType();
    auto func = dispatcher().lookup(input->device().getType());

    if (func == nullptr) {
        throw std::runtime_error("No Lp_Pool1d implementation found for device type: " + std::to_string(static_cast<int>(device_type)));
    }

    func(output, input, norm_type, kernel_size, stride, ceil_mode);
}

Tensor lp_pool1d(Tensor input, float norm_type, size_t kernel_size, size_t stride, bool ceil_mode) {
    auto ndim = input->ndim();
    auto input_shape = input->shape();

    if (ndim != 3 && ndim != 2) {
        throw std::runtime_error("Input tensor must be 3-dimensional (N, C, L_in) or (C, L_in)");
    }

    if (ndim == 2) {
        input = input->view({1, input_shape[0], input_shape[1]});
        input_shape = input->shape();
    }

    auto L_in = input_shape[2];
    size_t L_out = 0;
    if (ceil_mode) {
        L_out = static_cast<size_t>(std::ceil(static_cast<float>(L_in - kernel_size) / stride)) + 1;
    } else {
        L_out = static_cast<size_t>(std::floor(static_cast<float>(L_in - kernel_size) / stride)) + 1;
    }

    auto output_shape = Shape{input_shape[0], input_shape[1], L_out};

    auto output = Tensor::empty(output_shape, input->dtype(), input->device());
    lp_pool1d_(output, input, norm_type, kernel_size, stride, ceil_mode);
    return output;
}

void lp_pool1d_(Tensor output, Tensor input, float norm_type, size_t kernel_size, size_t stride, bool ceil_mode) {
    Lp_Pool1d::execute(output, input, norm_type, kernel_size, stride, ceil_mode);
}
} // namespace infinicore::op
