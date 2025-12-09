#include "infinicore/nn/functional.hpp"
#include "infinicore/ops.hpp"
#include <spdlog/spdlog.h>

namespace infinicore::nn {
namespace functional {

Tensor swiglu(const Tensor &up, const Tensor &gate) {
    // Delegate to InfiniCore op (backed by InfiniRT/InfiniOP)
    // Validation is handled by the op layer
    // output = up * gate * sigmoid(gate)
    return op::swiglu(up, gate);
}

} // namespace functional
} // namespace infinicore::nn
