#include "infinicore/nn/swiglu.hpp"
#include "infinicore/ops.hpp"
#include <spdlog/spdlog.h>
#include <stdexcept>

namespace infinicore::nn {

Tensor SwiGLU::forward(const Tensor &up, const Tensor &gate) const {
    // Validate inputs
    auto up_shape = up->shape();
    auto gate_shape = gate->shape();

    if (up_shape != gate_shape) {
        throw std::invalid_argument(
            "up and gate tensors must have the same shape. Got up=" + std::to_string(up_shape.back()) + ", gate=" + std::to_string(gate_shape.back()));
    }

    if (up->dtype() != gate->dtype()) {
        throw std::invalid_argument(
            "up and gate tensors must have the same dtype. Got up=" + std::to_string(static_cast<int>(up->dtype())) + ", gate=" + std::to_string(static_cast<int>(gate->dtype())));
    }

    // Delegate to InfiniCore op (backed by InfiniRT/InfiniOP)
    // output = up * gate * sigmoid(gate)
    // The op::swiglu function handles: out = up * gate * sigmoid(gate)
    return op::swiglu(up, gate);
}

std::string SwiGLU::extra_repr() const {
    return "SwiGLU()";
}

} // namespace infinicore::nn
