#include "infinicore/nn/linear.hpp"
#include "infinicore/ops.hpp"
#include <spdlog/spdlog.h>

namespace infinicore::nn {

Linear::Linear(size_t in_features, size_t out_features, bool bias, const DataType &dtype, const Device &device)
    : in_features_(in_features),
      out_features_(out_features),
      has_bias_(bias),
      dtype_(dtype) {

    device_ = device;

    // Initialize parameters using macro
    INFINICORE_NN_PARAMETER_INIT(weight, ({out_features, in_features}, dtype_, device));

    // Register bias parameter if requested
    if (bias) {
        INFINICORE_NN_PARAMETER_INIT(bias, ({out_features}, dtype_, device));
    } else {
        bias_ = Parameter(); // Default constructed empty parameter
    }

    SPDLOG_DEBUG("Created Linear module: in_features={}, out_features={}, bias={}, dtype={}",
                 in_features, out_features, bias, static_cast<int>(dtype_));
}

Tensor Linear::compute_linear(Tensor &input) const {
    // Create output tensor with shape [batch_size, out_features]
    auto output_shape = input->shape();
    output_shape[output_shape.size() - 1] = out_features_;
    auto output = Tensor::empty(output_shape, input->dtype(), input->device());

    // Transpose weight: [out_features, in_features] -> [in_features, out_features]
    auto weight_t = weight_->permute({1, 0});

    if (has_bias_) {
        // Broadcast bias to output shape
        size_t ndim_diff = output->ndim() - 1;
        std::vector<Stride> strides(ndim_diff, 0);
        strides.push_back(bias_->stride(0));
        auto bias_view = bias_->as_strided(output->shape(), strides);

        // Compute matmul result separately, then add to output
        infinicore::op::matmul_(output, input, weight_t);
        infinicore::op::add_(output, output, bias_view);
    } else {
        // No bias: just compute output = input @ weight_t
        infinicore::op::matmul_(output, input, weight_t);
    }

    return output;
}

Tensor Linear::forward(Tensor &input) const {
    return compute_linear(input);
}

Tensor Linear::forward(Tensor &input, Tensor &residual) const {
    auto output = compute_linear(input);

    // Add residual: output = output + residual
    infinicore::op::add_(output, output, residual);

    return output;
}

std::string Linear::extra_repr() const {
    return "Linear(in_features=" + std::to_string(in_features_) + ", out_features=" + std::to_string(out_features_) + ", bias=" + (has_bias_ ? "true" : "false") + ", dtype=" + std::to_string(static_cast<int>(dtype_)) + ")";
}

} // namespace infinicore::nn
