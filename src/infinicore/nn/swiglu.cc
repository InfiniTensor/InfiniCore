#include "infinicore/nn/swiglu.hpp"

namespace infinicore::nn {

Tensor SwiGLU::forward(const Tensor &up, const Tensor &gate) const {
    // Delegate to functional version
    return functional::swiglu(up, gate);
}

std::string SwiGLU::extra_repr() const {
    return "SwiGLU()";
}

} // namespace infinicore::nn
