#include "infinicore/ops/lp_pool2d.hpp"
#include <cmath>
#include <stdexcept>

namespace infinicore::op {

common::OpDispatcher<Lp_Pool2d::schema> &Lp_Pool2d::dispatcher() {
    static common::OpDispatcher<Lp_Pool2d::schema> dispatcher_;
    return dispatcher_;
};

void Lp_Pool2d::execute(Tensor output, Tensor input, float norm_type, const std::tuple<size_t, size_t> kernel_size, const std::tuple<size_t, size_t> stride, bool ceil_mode) {
    infinicore::context::setDevice(input->device(), true);
    auto device_type = context::getDevice().getType();
    auto func = dispatcher().lookup(device_type);

    if (func == nullptr) {
        throw std::runtime_error("No Lp_Pool2d implementation found for device type: " + std::to_string(static_cast<int>(device_type)));
    }

    func(output, input, norm_type, kernel_size, stride, ceil_mode);
}

Tensor lp_pool2d(Tensor input, float norm_type, const std::tuple<size_t, size_t> kernel_size, const std::tuple<size_t, size_t> stride, bool ceil_mode) {
    const auto ndim = input->ndim();
    auto input_shape = input->shape();

    if (ndim != 4 && ndim != 3) {
        throw std::runtime_error("Input tensor must be 4-dimensional (N, C, H_in, W_in) or (C, H_in, W_in)");
    }

    if (ndim == 3) {
        input = input->view({1, input_shape[0], input_shape[1], input_shape[2]});
        input_shape = input->shape();
    }

    const auto [Kernel_H, Kernel_W] = kernel_size;
    const auto [Stride_H, Stride_W] = stride;
    const auto H_in = input_shape[2];
    const auto W_in = input_shape[3];
    size_t H_out = 0;
    size_t W_out = 0;
    if (ceil_mode) {
        H_out = static_cast<size_t>(std::ceil(static_cast<float>(H_in - Kernel_H) / Stride_H)) + 1;
        W_out = static_cast<size_t>(std::ceil(static_cast<float>(W_in - Kernel_W) / Stride_W)) + 1;
    } else {
        H_out = static_cast<size_t>(std::floor(static_cast<float>(H_in - Kernel_H) / Stride_H)) + 1;
        W_out = static_cast<size_t>(std::floor(static_cast<float>(W_in - Kernel_W) / Stride_W)) + 1;
    }

    auto output_shape = Shape{input_shape[0], input_shape[1], H_out, W_out};

    auto output = Tensor::empty(output_shape, input->dtype(), input->device());
    lp_pool2d_(output, input, norm_type, kernel_size, stride, ceil_mode);
    return output;
}

void lp_pool2d_(Tensor output, Tensor input, float norm_type, const std::tuple<size_t, size_t> kernel_size, const std::tuple<size_t, size_t> stride, bool ceil_mode) {
    Lp_Pool2d::execute(output, input, norm_type, kernel_size, stride, ceil_mode);
}
} // namespace infinicore::op
