#include "infinicore/nn/linear.hpp"
#include "../utils.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/ops/linear.hpp"
#include <optional>
#include <spdlog/spdlog.h>

namespace infinicore::nn {

BaseLinear::BaseLinear(size_t in_features, size_t out_features, bool bias,
                       const DataType &dtype, const Device &device)
    : in_features_(in_features),
      out_features_(out_features),
      has_bias_(bias),
      dtype_(dtype) {

    device_ = device;
}

Tensor BaseLinear::compute_linear(Tensor &input) const {

    // Ensure input is contiguous before creating views (required for matmul)
    // This prevents hanging when input tensor has non-contiguous memory layout
    Tensor input_contiguous = input->is_contiguous() ? input : input->contiguous();

    // Use ops::linear_ directly to match Python backend's exact code path
    // This ensures identical computation and numerical results
    // Parameter inherits from Tensor, so we cast to Tensor explicitly
    Tensor weight_tensor = static_cast<const Tensor &>(weight_);
    std::optional<Tensor> bias_opt = has_bias_ ? std::make_optional<Tensor>(static_cast<const Tensor &>(bias_)) : std::nullopt;

    auto output = infinicore::op::linear(input_contiguous->contiguous(), weight_tensor->contiguous(), bias_opt);
    return output;
}

Tensor BaseLinear::forward(Tensor &input) const {
    return compute_linear(input);
}

Tensor BaseLinear::forward(Tensor &input, Tensor &residual) const {
    auto output = compute_linear(input);

    // Add residual: output = output + residual
    infinicore::op::add_(output, output, residual);

    return output;
}

} // namespace infinicore::nn

namespace infinicore::nn {

Linear::Linear(size_t in_features, size_t out_features, bool bias,
               const DataType &dtype, const Device &device)
    : BaseLinear(in_features, out_features, bias, dtype, device_) {

    device_ = device;

    // Initialize parameters using macro
    INFINICORE_NN_PARAMETER_INIT(weight, ({out_features, in_features}, dtype_, device));

    // Register bias parameter if requested
    if (bias) {
        INFINICORE_NN_PARAMETER_INIT(bias, ({out_features}, dtype_, device));
    } else {
        bias_ = Parameter(); // Default constructed empty parameter
    }

    // SPDLOG_DEBUG("Created Linear module: in_features={}, out_features={}, bias={}, dtype={}",
    //              in_features, out_features, bias, static_cast<int>(dtype_));
}

Tensor Linear::forward(Tensor &input) const {
    return BaseLinear::forward(input);
}

std::string Linear::extra_repr() const {
    return "Linear(in_features=" + std::to_string(in_features_) + ", out_features=" + std::to_string(out_features_) + ", bias=" + (has_bias_ ? "true" : "false") + ", dtype=" + std::to_string(static_cast<int>(dtype_)) + ")";
}

} // namespace infinicore::nn

namespace infinicore::nn {

ColumnParallelLinear::ColumnParallelLinear(size_t in_features, size_t out_features, bool bias,
                                           const DataType &dtype, const Device &device,
                                           Size tp_rank, Size tp_size)
    : BaseLinear(in_features, out_features, bias, dtype, device_),
      tp_rank_(tp_rank),
      tp_size_(tp_size) {

    device_ = device;

    // Initialize parameters using macro
    INFINICORE_NN_PARAMETER_INIT(weight, ({out_features, in_features}, dtype_, device,
                                          0, tp_rank_, tp_size_));

    // Register bias parameter if requested
    if (bias) {
        INFINICORE_NN_PARAMETER_INIT(bias, ({out_features}, dtype_, device,
                                            0, tp_rank_, tp_size_));
    } else {
        bias_ = Parameter(); // Default constructed empty parameter
    }

    // SPDLOG_DEBUG("Created ColumnParallelLinear module: in_features={}, out_features={}, bias={}, dtype={}",
    //              in_features, out_features, bias, static_cast<int>(dtype_));
}

Tensor ColumnParallelLinear::forward(Tensor &input) const {
    return BaseLinear::forward(input);
}

std::string ColumnParallelLinear::extra_repr() const {
    return "ColumnParallelLinear(in_features=" + std::to_string(in_features_) + ", out_features=" + std::to_string(out_features_) + ", bias=" + (has_bias_ ? "true" : "false") + ", dtype=" + std::to_string(static_cast<int>(dtype_)) + ")";
}

} // namespace infinicore::nn

namespace infinicore::nn {

RowParallelLinear::RowParallelLinear(size_t in_features, size_t out_features, bool bias,
                                     const DataType &dtype, const Device &device,
                                     Size tp_rank, Size tp_size, infinicclComm_t communicator)
    : BaseLinear(in_features, out_features, bias, dtype, device_),
      tp_rank_(tp_rank),
      tp_size_(tp_size), communicator_(communicator) {

    device_ = device;

    // Initialize parameters using macro
    INFINICORE_NN_PARAMETER_INIT(weight, ({out_features, in_features}, dtype_, device,
                                          1, tp_rank_, tp_size_));

    // Register bias parameter if requested
    if (bias && (0 == tp_rank_)) {
        INFINICORE_NN_PARAMETER_INIT(bias, ({out_features}, dtype_, device, 0, 0, 1));
    } else {
        bias_ = Parameter(); // Default constructed empty parameter
    }

    // SPDLOG_DEBUG("Created RowParallelLinear module: in_features={}, out_features={}, bias={}, dtype={}",
    //              in_features, out_features, bias, static_cast<int>(dtype_));
}

Tensor RowParallelLinear::forward(Tensor &input) const {
    auto output = BaseLinear::forward(input);

    if ((tp_size_ > 1) && (communicator_ != nullptr)) {

        Size count = output->numel();
        DataType type = output->dtype();

        infinirtStream_t stream = infinicore::context::getStream();

        INFINICORE_CHECK_ERROR(infinicclAllReduce(output->data(), output->data(), count, static_cast<infiniDtype_t>(static_cast<int>(type)),
                                                  INFINICCL_SUM, communicator_, stream));
        INFINICORE_CHECK_ERROR(infinirtStreamSynchronize(stream));

        // RUN_INFINI(infinirtStreamSynchronize(stream));
    }
    return output;
}

std::string RowParallelLinear::extra_repr() const {
    return "RowParallelLinear(in_features=" + std::to_string(in_features_) + ", out_features=" + std::to_string(out_features_) + ", bias=" + (has_bias_ ? "true" : "false") + ", dtype=" + std::to_string(static_cast<int>(dtype_)) + ")";
}

} // namespace infinicore::nn
