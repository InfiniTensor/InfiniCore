#include "infinicore/nn/linear.hpp"
#include "../utils.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/ops/linear.hpp"
#include "infinicore/ops/linear_w8a8i8.hpp"
#include <optional>
#include <spdlog/spdlog.h>

#include <iostream>

namespace infinicore::nn {

BaseLinear::BaseLinear(size_t in_features, size_t out_features, bool bias,
                       const DataType &dtype, const Device &device,
                       const std::optional<QuantConfig> &quant_config)
    : in_features_(in_features),
      out_features_(out_features),
      has_bias_(bias),
      dtype_(dtype),
      quant_config_(quant_config) {

    device_ = device;
}

Tensor BaseLinear::compute_linear(Tensor &input) const {
    if (!this->is_quantized()) {
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
    } else {
        switch (this->get_quant_scheme()) {
        case infinicore::nn::QuantScheme::COMPRESSED_TENSOR_W8A8I8: {
            Tensor input_contiguous = input->is_contiguous() ? input : input->contiguous();
            // input_contiguous = input_contiguous->view({input_contiguous->shape()[1], input_contiguous->shape()[2]});
            Tensor weight_packed_tensor = static_cast<const Tensor &>(weight_);
            Tensor weight_scale_tensor = static_cast<const Tensor &>(weight_scale_);
            // weight_packed should be transposed and non-contiguous.
            std::optional<Tensor> bias_opt = has_bias_ ? std::make_optional<Tensor>(static_cast<const Tensor &>(bias_)) : std::nullopt;

            auto output = infinicore::op::linear_w8a8i8(input_contiguous->contiguous(), weight_packed_tensor, weight_scale_tensor, bias_opt);
            return output;
        }
        default: {
            // Temp for test
            return input;
        }
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
               const DataType &dtype, const Device &device,
               const std::optional<QuantConfig> &quant_config)
    : BaseLinear(in_features, out_features, bias, dtype, device_, quant_config) {

    device_ = device;

    if (!this->is_quantized()) {
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
    } else {
        switch (this->get_quant_scheme()) {
        case infinicore::nn::QuantScheme::COMPRESSED_TENSOR_W8A8I8: {
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
            break;
        }
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
                                           Size tp_rank, Size tp_size,
                                           const std::optional<QuantConfig> &quant_config)
    : BaseLinear(in_features, out_features, bias, dtype, device_, quant_config),
      tp_rank_(tp_rank),
      tp_size_(tp_size) {

    device_ = device;

    if (!this->is_quantized()) {
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
    } else {
        switch (this->get_quant_scheme()) {
        case infinicore::nn::QuantScheme::COMPRESSED_TENSOR_W8A8I8: {

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
            break;
        }
        }
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
                                     Size tp_rank, Size tp_size, infinicclComm_t communicator,
                                     const std::optional<QuantConfig> &quant_config)
    : BaseLinear(in_features, out_features, bias, dtype, device_, quant_config),
      tp_rank_(tp_rank),
      tp_size_(tp_size), communicator_(communicator) {

    device_ = device;
    if (!this->is_quantized()) {
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
    } else {
        switch (this->get_quant_scheme()) {
        case infinicore::nn::QuantScheme::COMPRESSED_TENSOR_W8A8I8: {
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
            break;
        }
        }
    }
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
