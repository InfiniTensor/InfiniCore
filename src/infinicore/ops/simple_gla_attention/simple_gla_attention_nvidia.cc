#include "infinicore/ops/simple_gla_attention.hpp"

#include "../../utils.hpp"
#include "infinicore/ops/simple_gla_prefill.hpp"

namespace infinicore::op {

namespace {

// Prefer the `simple_gla_prefill` implementation (InfiniOP-backed) on NVIDIA.
void simple_gla_attention_nvidia_calculate(Tensor &out,
                                           const Tensor &q,
                                           const Tensor &k,
                                           const Tensor &v,
                                           const Tensor &g_gamma,
                                           float scale) {
    SimpleGLAPrefill::execute(out, q, k, v, g_gamma, scale);
}

static bool register_nvidia = []() {
    SimpleGlaAttention::dispatcher().registerDevice(
        Device::Type::NVIDIA,
        &simple_gla_attention_nvidia_calculate,
        false);
    return true;
}();

} // namespace

} // namespace infinicore::op
