#include "infinicore/nn/linear.hpp"
#include "../utils.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/ops/distributed/allreduce.hpp"
#include "infinicore/ops/linear.hpp"
#include "infinicore/ops/linear_w8a8i8.hpp"
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

BaseLinear::BaseLinear(size_t in_features, size_t out_features, std::shared_ptr<infinicore::quantization::BaseQuantization> quantization, bool bias,
                       const DataType &dtype, const Device &device)
    : in_features_(in_features),
      out_features_(out_features),
      quantization_(quantization),
      has_bias_(bias),
      dtype_(dtype) {

    device_ = device;
}

Tensor BaseLinear::compute_linear(Tensor &input) const {
    switch (this->quantization_->get_quant_scheme()) {
    case infinicore::quantization::QuantScheme::COMPRESSED_TENSOR_W8A8I8: {
        Tensor input_contiguous = input->is_contiguous() ? input : input->contiguous();

        Tensor weight_packed_tensor = static_cast<const Tensor &>(weight_);
        Tensor weight_scale_tensor = static_cast<const Tensor &>(weight_scale_);
        // weight_packed should be transposed and non-contiguous.
        std::optional<Tensor> bias_opt = has_bias_ ? std::make_optional<Tensor>(static_cast<const Tensor &>(bias_)) : std::nullopt;

        auto output = infinicore::op::linear_w8a8i8(input_contiguous->contiguous(), weight_packed_tensor, weight_scale_tensor, bias_opt);
        return output;
    }
    default: {
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
    }
} // namespace infinicore::nn

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

Linear::Linear(size_t in_features, size_t out_features,
               std::shared_ptr<infinicore::quantization::BaseQuantization> quantization, bool bias,
               const DataType &dtype, const Device &device)
    : BaseLinear(in_features, out_features, quantization, bias, dtype, device_) {

    device_ = device;

    switch (this->quantization_->get_quant_scheme()) {
    case infinicore::quantization::QuantScheme::COMPRESSED_TENSOR_W8A8I8: {
        INFINICORE_NN_PARAMETER_INIT(weight, ({out_features, in_features}, infinicore::DataType::I8, device));
        INFINICORE_NN_PARAMETER_INIT(weight_scale, ({out_features, 1}, infinicore::DataType::F32, device));

        if (bias) {
            INFINICORE_NN_PARAMETER_INIT(bias, ({out_features}, dtype_, device));
        } else {
            bias_ = Parameter();
        }
        break;
    }
    default: {
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
        break;
    }
    }
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
}

ColumnParallelLinear::ColumnParallelLinear(size_t in_features, size_t out_features, std::shared_ptr<infinicore::quantization::BaseQuantization> quantization, bool bias,
                                           const DataType &dtype, const Device &device,
                                           Size tp_rank, Size tp_size)
    : BaseLinear(in_features, out_features, quantization, bias, dtype, device_),
      tp_rank_(tp_rank),
      tp_size_(tp_size) {

    device_ = device;

    switch (this->quantization_->get_quant_scheme()) {
    case infinicore::quantization::QuantScheme::COMPRESSED_TENSOR_W8A8I8: {

        INFINICORE_NN_PARAMETER_INIT(weight, ({out_features, in_features}, infinicore::DataType::I8, device, 0, tp_rank_, tp_size_));
        INFINICORE_NN_PARAMETER_INIT(weight_scale, ({out_features, 1}, infinicore::DataType::F32, device, 0, tp_rank_, tp_size_));

        if (bias) {
            INFINICORE_NN_PARAMETER_INIT(bias, ({out_features}, dtype_, device, 0, 0, 1));
        } else {
            bias_ = Parameter();
        }
        break;
    }
    default: {
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
        break;
    }
    }
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
}

RowParallelLinear::RowParallelLinear(size_t in_features, size_t out_features, std::shared_ptr<infinicore::quantization::BaseQuantization> quantization, bool bias,
                                     const DataType &dtype, const Device &device,
                                     Size tp_rank, Size tp_size, infinicclComm_t communicator)
    : BaseLinear(in_features, out_features, quantization, bias, dtype, device_),
      tp_rank_(tp_rank),
      tp_size_(tp_size), communicator_(communicator) {

    device_ = device;

    switch (this->quantization_->get_quant_scheme()) {
    case infinicore::quantization::QuantScheme::COMPRESSED_TENSOR_W8A8I8: {
        INFINICORE_NN_PARAMETER_INIT(weight, ({out_features, in_features}, infinicore::DataType::I8, device, 1, tp_rank_, tp_size_));
        INFINICORE_NN_PARAMETER_INIT(weight_scale, ({out_features, 1}, infinicore::DataType::F32, device, 0, 0, 1));

        if (bias) {
            INFINICORE_NN_PARAMETER_INIT(bias, ({out_features}, dtype_, device, 0, tp_rank_, tp_size_));
        } else {
            bias_ = Parameter();
        }
        break;
    }
    default: {
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
        break;
    }
    }
}

Tensor RowParallelLinear::forward(Tensor &input) const {
    auto output = BaseLinear::forward(input);

    if ((tp_size_ > 1) && (communicator_ != nullptr)) {
        op::distributed::allreduce_(output, output, INFINICCL_SUM, communicator_);
    }
    return output;
}

std::string RowParallelLinear::extra_repr() const {
    return "RowParallelLinear(in_features=" + std::to_string(in_features_) + ", out_features=" + std::to_string(out_features_) + ", bias=" + (has_bias_ ? "true" : "false") + ", dtype=" + std::to_string(static_cast<int>(dtype_)) + ")";
}

} // namespace infinicore::nn
