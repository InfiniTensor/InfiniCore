#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/ops.hpp"
#include <cmath>
#include <stdexcept>

namespace infinicore::nn {

RMSNorm::RMSNorm(size_t normalized_shape, double eps, const DataType &dtype, const Device &device)
    : normalized_shape_(normalized_shape),
      eps_(eps),
      dtype_(dtype) {

    device_ = device;

    // Initialize parameter using macro
    INFINICORE_NN_PARAMETER_INIT(weight, ({normalized_shape}, dtype_, device));

    // Initialize weight to ones (standard practice for RMSNorm)
    auto ones_tensor = Tensor::ones({normalized_shape}, dtype_, device);
    weight_->copy_from(ones_tensor);
}

Tensor RMSNorm::forward(const Tensor &x) const {
    // Delegate to InfiniCore op (backed by InfiniRT/InfiniOP)
    // Validation is handled by the op layer
    return op::rms_norm(x, weight_, static_cast<float>(eps_));
}

std::string RMSNorm::extra_repr() const {
    return "RMSNorm(normalized_shape=" + std::to_string(normalized_shape_) + ", eps=" + std::to_string(eps_) + ", dtype=" + std::to_string(static_cast<int>(dtype_)) + ")";
}

} // namespace infinicore::nn
