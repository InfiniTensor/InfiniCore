#pragma once

#include "../tensor.hpp"

namespace infinicore::nn {

/**
 * @brief Functional operations namespace
 *
 * Similar to torch.nn.functional, this namespace provides stateless functional operations
 * that don't require module instantiation. These are pure functions that operate on tensors.
 *
 * Use functional operations when:
 * - The operation has no learnable parameters
 * - The operation has no internal state or buffers
 * - You want lightweight, stateless operations
 *
 * For operations with parameters or state, use the corresponding Module classes.
 */
namespace functional {

/**
 * @brief SwiGLU activation function
 *
 * Applies SwiGLU (Swish-Gated Linear Unit) activation: output = up * gate * sigmoid(gate)
 *
 * This is the functional interface for SwiGLU. The module version (nn::SwiGLU) wraps
 * this function. Since SwiGLU has no parameters or buffers, you can use either:
 * - Functional: output = functional::swiglu(up, gate);
 * - Module: SwiGLU swiglu; output = swiglu.forward(up, gate);
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
Tensor swiglu(const Tensor &up, const Tensor &gate);

} // namespace functional

} // namespace infinicore::nn
