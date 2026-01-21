#include "infinicore/nn/layernorm.hpp"

namespace infinicore::nn {

LayerNorm::LayerNorm(size_t normalized_shape,
                     double eps,
                     const DataType &dtype,
                     const Device &device)
    : normalized_shape_(normalized_shape),
      eps_(eps),
      dtype_(dtype) {
    INFINICORE_NN_PARAMETER_INIT(weight, ({normalized_shape_}, dtype_, device));
    INFINICORE_NN_PARAMETER_INIT(bias, ({normalized_shape_}, dtype_, device));
    auto weight_init = infinicore::Tensor::ones({normalized_shape_}, dtype_, device);
    auto bias_init = infinicore::Tensor::zeros({normalized_shape_}, dtype_, device);
    weight_->copy_from(weight_init);
    bias_->copy_from(bias_init);
}

Tensor LayerNorm::forward(const Tensor &x) const {
    return infinicore::op::layer_norm(x, weight_, bias_, static_cast<float>(eps_));
}

std::string LayerNorm::extra_repr() const {
    return "normalized_shape=" + std::to_string(normalized_shape_) +
           ", eps=" + std::to_string(eps_) +
           ", dtype=" + infinicore::toString(dtype_);
}

} // namespace infinicore::nn
