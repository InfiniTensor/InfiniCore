#pragma once

#include "../device.hpp"
#include "../tensor.hpp"
#include "common/dispatcher.hpp"

namespace infinicore::op {

// Batched update of Simple GLA recurrent state (float32 [B,H,D,D]) for a contiguous
// K/V segment [B,L,H,D], matching L repeated simple_gla_decode_step applications:
//   S <- g^L * S + sum_{j=0}^{L-1} g^{L-1-j} * outer(k_j, v_j)
// g_gamma: [H] (same log-gate as simple_gla_decode_step; gate = exp(g_gamma)).
class SimpleGlaRecurrentStateAppend {
public:
    using schema = void (*)(Tensor &state, const Tensor &k_seg, const Tensor &v_seg, const Tensor &g_gamma);
    static void execute(Tensor &state, const Tensor &k_seg, const Tensor &v_seg, const Tensor &g_gamma);
    static common::OpDispatcher<schema> &dispatcher();
};

void simple_gla_recurrent_state_append_segment(Tensor &state, const Tensor &k_seg, const Tensor &v_seg,
                                             const Tensor &g_gamma);

} // namespace infinicore::op
