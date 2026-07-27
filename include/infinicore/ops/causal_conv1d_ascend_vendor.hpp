#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"
#include "infinicore.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace infinicore::op {

// Ascend vendor fused causal-conv bridge. The vendor kernel consumes the
// vLLM-Ascend layout directly: x/out [tokens, C], weight [K, C], and state
// [pool, K - 1, C]. It also fuses SiLU and state-cache updates.
INFINICORE_GRAPH_OP_CLASS(
    CausalConv1dAscendVendor,
    Tensor,
    Tensor,
    const Tensor &,
    const Tensor &,
    std::optional<Tensor>,
    std::vector<int64_t>,
    std::vector<int64_t>,
    bool,
    bool);

__export Tensor causal_conv1d_ascend_vendor(
    const Tensor &x,
    Tensor conv_state,
    const Tensor &weight,
    std::optional<Tensor> bias,
    std::vector<int64_t> query_start_loc,
    std::vector<int64_t> cache_indices,
    bool fuse_silu,
    bool decode);

} // namespace infinicore::op
