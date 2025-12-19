#include "infinicore/ops/fold.hpp"
#include <cmath>
#include <stdexcept>

namespace infinicore::op {

common::OpDispatcher<Fold::schema> &Fold::dispatcher() {
    static common::OpDispatcher<Fold::schema> dispatcher_;
    return dispatcher_;
};

void Fold::execute(Tensor output, Tensor input, std::tuple<size_t, size_t> output_size, std::tuple<size_t, size_t> kernel_size, std::tuple<size_t, size_t> dilation, std::tuple<size_t, size_t> padding, std::tuple<size_t, size_t> stride) {
    infinicore::context::setDevice(input->device(), true);
    auto device_type = context::getDevice().getType();
    auto func = dispatcher().lookup(device_type);

    if (func == nullptr) {
        throw std::runtime_error("No Fold implementation found for device type: " + std::to_string(static_cast<int>(device_type)));
    }

    func(output, input, output_size, kernel_size, dilation, padding, stride);
}

Tensor fold(Tensor input, std::tuple<size_t, size_t> output_size, std::tuple<size_t, size_t> kernel_size, std::tuple<size_t, size_t> dilation, std::tuple<size_t, size_t> padding, std::tuple<size_t, size_t> stride) {
    const auto ndim = input->ndim();
    auto input_shape = input->shape();

    if (ndim != 3 && ndim != 2) {
        throw std::runtime_error("Input tensor must be 3-dimensional (N, C * K_h * K_w, L) or 2-dimensional (C * K_h * K_w, L)");
    }

    // Normalize input to shape [N, C*K_h*K_w, L]
    if (ndim == 2) {
        // input: [C*K_h*K_w, L] -> [1, C*K_h*K_w, L]
        input = input->view({1, input_shape[0], input_shape[1]});
        input_shape = input->shape();
    } // if ndim==3, assume already [N, C*K*K, L]

    // input: [N, C * K_h * K_w, L]
    const auto [Kernel_H, Kernel_W] = kernel_size;
    const auto [Output_H, Output_W] = output_size;
    const auto [Dilation_H, Dilation_W] = dilation;
    const auto [Padding_H, Padding_W] = padding;
    const auto [Stride_H, Stride_W] = stride;
    const auto C = input_shape[1] / (Kernel_H * Kernel_W);

    if (C * Kernel_H * Kernel_W != input_shape[1]) {
        throw std::runtime_error("Input channel dimension is not divisible by kernel size product");
    }
    // Validate input L equals computed number of sliding positions
    const auto L = input_shape[2];
    const auto L_h = (Output_H + 2 * Padding_H >= Dilation_H * (Kernel_H - 1) + 1)
                         ? (static_cast<size_t>(std::floor((static_cast<double>(Output_H) + 2.0 * Padding_H - static_cast<double>(Dilation_H) * (Kernel_H - 1) - 1) / Stride_H)) + 1)
                         : 0;
    const auto L_w = (Output_W + 2 * Padding_W >= Dilation_W * (Kernel_W - 1) + 1)
                         ? (static_cast<size_t>(std::floor((static_cast<double>(Output_W) + 2.0 * Padding_W - static_cast<double>(Dilation_W) * (Kernel_W - 1) - 1) / Stride_W)) + 1)
                         : 0;
    if (L != L_h * L_w) {
        throw std::runtime_error("Input L does not match computed sliding window count");
    }

    auto output_shape = Shape{input_shape[0], C, Output_H, Output_W};

    auto output = Tensor::empty(output_shape, input->dtype(), input->device());
    fold_(output, input, output_size, kernel_size, dilation, padding, stride);
    return output;
}

void fold_(Tensor output, Tensor input, std::tuple<size_t, size_t> output_size, std::tuple<size_t, size_t> kernel_size, std::tuple<size_t, size_t> dilation, std::tuple<size_t, size_t> padding, std::tuple<size_t, size_t> stride) {
    Fold::execute(output, input, output_size, kernel_size, dilation, padding, stride);
}
} // namespace infinicore::op
