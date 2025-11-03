#include "infinicore/nn/swiglu.hpp"
#include "infinicore/ops.hpp"
#include <spdlog/spdlog.h>
#include <stdexcept>

namespace infinicore::nn {

Tensor SwiGLU::forward(const Tensor &up, const Tensor &gate) const {
    // Delegate to InfiniCore op (backed by InfiniRT/InfiniOP)
    // Validation is handled by the op layer
    // output = up * gate * sigmoid(gate)
    return op::swiglu(up, gate);
}

std::string SwiGLU::extra_repr() const {
    return "SwiGLU()";
}

} // namespace infinicore::nn
