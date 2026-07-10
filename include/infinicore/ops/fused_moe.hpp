#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"
#include <optional>

namespace infinicore::op {

enum class FusedMoeActivation : int {
    Silu = 0,
    Swiglu = 1,
};

INFINICORE_GRAPH_OP_CLASS(FusedMoe,
                          Tensor,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          std::optional<Tensor>,
                          std::optional<Tensor>,
                          FusedMoeActivation);

Tensor fused_moe(const Tensor &input,
                 const Tensor &token_selected_experts,
                 const Tensor &token_final_scales,
                 const Tensor &w1,
                 const Tensor &w2,
                 std::optional<Tensor> b1,
                 std::optional<Tensor> b2,
                 FusedMoeActivation activation);

void fused_moe_(Tensor out,
                const Tensor &input,
                const Tensor &token_selected_experts,
                const Tensor &token_final_scales,
                const Tensor &w1,
                const Tensor &w2,
                std::optional<Tensor> b1,
                std::optional<Tensor> b2,
                FusedMoeActivation activation);

} // namespace infinicore::op
