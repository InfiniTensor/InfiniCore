#pragma once

#include "module.hpp"
#include "functional.hpp"

namespace infinicore::nn {

/**
 * @brief SwiGLU module
 *
 * Applies SwiGLU (Swish-Gated Linear Unit) activation: output = up * gate * sigmoid(gate)
 *
 * This module wraps the functional::swiglu() operation. Since SwiGLU has no parameters
 * or buffers, you can use either the module or the functional version:
 * - Module: SwiGLU swiglu; output = swiglu.forward(up, gate);
 * - Functional: output = functional::swiglu(up, gate);
 */
class SwiGLU : public Module {
public:
    /**
     * @brief Construct a SwiGLU module
     *
     * SwiGLU is a stateless activation function, so no parameters are needed.
     */
    SwiGLU() = default;

    /**
     * @brief Forward pass: apply SwiGLU activation
     *
     * @param up The "up" projection tensor
     * @param gate The "gate" projection tensor
     * @return Result tensor: up * gate * sigmoid(gate)
     *
     * Both input tensors must have the same shape and dtype.
     * Common usage:
     *   - Input: up from linear_up layer, gate from linear_gate layer
     *   - Shapes: typically [batch, seq_len, hidden_size] or [batch, hidden_size]
     */
    Tensor forward(const Tensor &up, const Tensor &gate) const;

    // String representation
    std::string extra_repr() const;
};

} // namespace infinicore::nn
