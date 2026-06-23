#include "infinicore/ops/conv1d.hpp"

#include "../../utils.hpp"

#include <stdexcept>

namespace infinicore::op {

common::OpDispatcher<Conv1d::schema> &Conv1d::dispatcher() {
    static common::OpDispatcher<Conv1d::schema> dispatcher_;
    return dispatcher_;
}

void Conv1d::execute(Tensor output,
                     Tensor input,
                     Tensor weight,
                     Tensor bias,
                     const size_t *pads,
                     const size_t *strides,
                     const size_t *dilations,
                     size_t n) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input, weight);
    if (bias) {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, bias);
    }
    infinicore::context::setDevice(output->device());
    auto device_type = output->device().getType();
    auto func = dispatcher().lookup(device_type);

    if (func == nullptr) {
        throw std::runtime_error("No Conv1d implementation found for device type: " + std::to_string(static_cast<int>(device_type)));
    }

    func(output, input, weight, bias, pads, strides, dilations, n);
}

static size_t conv1d_out_size(size_t input, size_t padding, size_t dilation, size_t kernel, size_t stride) {
    if (stride == 0 || dilation == 0 || kernel == 0) {
        throw std::runtime_error("conv1d: stride, dilation, and kernel size must be greater than zero");
    }
    size_t effective_kernel = dilation * (kernel - 1) + 1;
    size_t padded_input = input + 2 * padding;
    if (padded_input < effective_kernel) {
        throw std::runtime_error("Invalid conv1d output shape (negative or zero)");
    }
    return (padded_input - effective_kernel) / stride + 1;
}

static void validate_conv1d_shapes(Tensor output,
                                   Tensor input,
                                   Tensor weight,
                                   std::optional<Tensor> bias,
                                   size_t groups) {
    const auto &out_shape = output->shape();
    const auto &in_shape = input->shape();
    const auto &w_shape = weight->shape();

    if (in_shape.size() != 3 || w_shape.size() != 3 || out_shape.size() != 3) {
        throw std::runtime_error("conv1d expects input [N, C_in, L], weight [C_out, C_in/groups, K], and output [N, C_out, L_out]");
    }
    if (groups == 0) {
        throw std::runtime_error("conv1d: groups must be greater than zero");
    }
    if (in_shape[1] % groups != 0 || w_shape[0] % groups != 0) {
        throw std::runtime_error("conv1d: input channels and output channels must be divisible by groups");
    }
    if (w_shape[1] != in_shape[1] / groups) {
        throw std::runtime_error("conv1d: weight input channels must equal input channels divided by groups");
    }
    if (out_shape[0] != in_shape[0] || out_shape[1] != w_shape[0]) {
        throw std::runtime_error("conv1d: output batch or channel dimension is invalid");
    }
    if (bias) {
        const auto &b_shape = (*bias)->shape();
        if (b_shape.size() != 1 || b_shape[0] != w_shape[0]) {
            throw std::runtime_error("conv1d: bias must have shape [C_out]");
        }
    }
}

Tensor conv1d(Tensor input,
              Tensor weight,
              std::optional<Tensor> bias,
              size_t stride,
              size_t padding,
              size_t dilation,
              size_t groups) {
    const auto &in_shape = input->shape();
    const auto &w_shape = weight->shape();
    if (in_shape.size() != 3 || w_shape.size() != 3) {
        throw std::runtime_error("conv1d expects input [N, C_in, L] and weight [C_out, C_in/groups, K]");
    }

    size_t l_out = conv1d_out_size(in_shape[2], padding, dilation, w_shape[2], stride);
    Shape out_shape = {in_shape[0], w_shape[0], l_out};

    auto output = Tensor::empty(out_shape, input->dtype(), input->device());
    conv1d_(output, input, weight, bias, stride, padding, dilation, groups);
    return output;
}

void conv1d_(Tensor output,
             Tensor input,
             Tensor weight,
             std::optional<Tensor> bias,
             size_t stride,
             size_t padding,
             size_t dilation,
             size_t groups) {
    validate_conv1d_shapes(output, input, weight, bias, groups);

    size_t expected_l_out = conv1d_out_size(input->shape()[2], padding, dilation, weight->shape()[2], stride);
    if (output->shape()[2] != expected_l_out) {
        throw std::runtime_error("conv1d: output length is invalid");
    }

    size_t in_channels_per_group = input->shape()[1] / groups;
    size_t out_channels_per_group = weight->shape()[0] / groups;

    for (size_t group = 0; group < groups; ++group) {
        Tensor group_input = groups == 1
                               ? input
                               : input->narrow({{1, group * in_channels_per_group, in_channels_per_group}})->contiguous();
        Tensor group_weight = groups == 1
                                ? weight
                                : weight->narrow({{0, group * out_channels_per_group, out_channels_per_group}});
        Tensor group_output = groups == 1
                                ? output
                                : Tensor::empty({output->shape()[0], out_channels_per_group, output->shape()[2]},
                                                output->dtype(),
                                                output->device());
        Tensor group_bias;
        if (bias) {
            group_bias = groups == 1
                           ? *bias
                           : (*bias)->narrow({{0, group * out_channels_per_group, out_channels_per_group}});
        }

        Conv1d::execute(group_output,
                        group_input,
                        group_weight,
                        group_bias,
                        &padding,
                        &stride,
                        &dilation,
                        1);

        if (groups != 1) {
            output->narrow({{1, group * out_channels_per_group, out_channels_per_group}})
                ->copy_from(group_output);
        }
    }
}
} // namespace infinicore::op
