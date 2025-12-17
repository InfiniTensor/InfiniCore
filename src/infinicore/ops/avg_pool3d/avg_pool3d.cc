#include "infinicore/ops/avg_pool3d.hpp"
#include <cmath>
#include <stdexcept>

namespace infinicore::op {

common::OpDispatcher<AvgPool3d::schema> &AvgPool3d::dispatcher() {
    static common::OpDispatcher<AvgPool3d::schema> dispatcher_;
    return dispatcher_;
};

void AvgPool3d::execute(Tensor output, Tensor input, std::tuple<size_t, size_t, size_t> kernel_size, std::tuple<size_t, size_t, size_t> stride, std::tuple<size_t, size_t, size_t> padding, bool ceil_mode) {
    infinicore::context::setDevice(input->device(), true);
    auto device_type = context::getDevice().getType();
    auto func = dispatcher().lookup(device_type);

    if (func == nullptr) {
        throw std::runtime_error("No AvgPool3d implementation found for device type: " + std::to_string(static_cast<int>(device_type)));
    }

    func(output, input, kernel_size, stride, padding, ceil_mode);
}

Tensor avg_pool3d(Tensor input, std::tuple<size_t, size_t, size_t> kernel_size, std::tuple<size_t, size_t, size_t> stride, std::tuple<size_t, size_t, size_t> padding, bool ceil_mode) {
    const auto ndim = input->ndim();
    auto input_shape = input->shape();

    if (ndim != 5 && ndim != 4) {
        throw std::runtime_error("Input tensor must be 4-dimensional (N, C, D_in, H_in, W_in) or (C, D_in, H_in, W_in)");
    }

    if (ndim == 4) {
        input = input->view({1, input_shape[0], input_shape[1], input_shape[2], input_shape[3]});
        input_shape = input->shape();
    }

    const auto [Kernel_D, Kernel_H, Kernel_W] = kernel_size;
    const auto [Stride_D, Stride_H, Stride_W] = stride;
    const auto [Padding_D, Padding_H, Padding_W] = padding;
    const auto D_in = input_shape[2];
    const auto H_in = input_shape[3];
    const auto W_in = input_shape[4];
    size_t D_out = 0;
    size_t H_out = 0;
    size_t W_out = 0;
    if (ceil_mode) {
        D_out = static_cast<size_t>(std::ceil(static_cast<float>(D_in + 2 * Padding_D - Kernel_D) / Stride_D)) + 1;
        H_out = static_cast<size_t>(std::ceil(static_cast<float>(H_in + 2 * Padding_H - Kernel_H) / Stride_H)) + 1;
        W_out = static_cast<size_t>(std::ceil(static_cast<float>(W_in + 2 * Padding_W - Kernel_W) / Stride_W)) + 1;
    } else {
        D_out = static_cast<size_t>(std::floor(static_cast<float>(D_in + 2 * Padding_D - Kernel_D) / Stride_D)) + 1;
        H_out = static_cast<size_t>(std::floor(static_cast<float>(H_in + 2 * Padding_H - Kernel_H) / Stride_H)) + 1;
        W_out = static_cast<size_t>(std::floor(static_cast<float>(W_in + 2 * Padding_W - Kernel_W) / Stride_W)) + 1;
    }

    auto output_shape = Shape{input_shape[0], input_shape[1], D_out, H_out, W_out};

    auto output = Tensor::empty(output_shape, input->dtype(), input->device());
    avg_pool3d_(output, input, kernel_size, stride, padding, ceil_mode);
    return output;
}

void avg_pool3d_(Tensor output, Tensor input, std::tuple<size_t, size_t, size_t> kernel_size, std::tuple<size_t, size_t, size_t> stride, std::tuple<size_t, size_t, size_t> padding, bool ceil_mode) {
    AvgPool3d::execute(output, input, kernel_size, stride, padding, ceil_mode);
}
} // namespace infinicore::op
