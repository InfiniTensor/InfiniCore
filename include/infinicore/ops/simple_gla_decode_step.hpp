#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {

// One decode timestep of Simple GLA (same recurrence as SimpleGlaAttention).
// q, k, v: [B, 1, H, D]; g_gamma: [H] (log-decay per head); state: [B, H, D, D] float32 (in-place).
// Updates: state = state * exp(g_gamma) + outer(k, v); then out[b,0,h,:] = (q * scale) @ state[b,h].
// Returns out with shape [B, 1, H, D] (same dtype as q).
class SimpleGlaDecodeStep {
public:
    using schema = void (*)(Tensor &out, Tensor &state, const Tensor &q, const Tensor &k, const Tensor &v,
                            const Tensor &g_gamma, float scale);
    static void execute(Tensor &out, Tensor &state, const Tensor &q, const Tensor &k, const Tensor &v,
                        const Tensor &g_gamma, float scale);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor simple_gla_decode_step(const Tensor &q, const Tensor &k, const Tensor &v, Tensor &state,
                              const Tensor &g_gamma, float scale);

} // namespace infinicore::op
