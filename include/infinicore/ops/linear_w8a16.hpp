#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"
#include <optional>

namespace infinicore::op {

// Weight-only int8 GEMM. Activation remains F16/BF16.
// input: [M, in_features]; qweight: [out_features, in_features];
// scale: [1, out_features], same dtype as input.
INFINICORE_GRAPH_OP_CLASS(
    LinearW8A16,
    Tensor,
    const Tensor &,
    const Tensor &,
    const Tensor &,
    std::optional<Tensor>);

Tensor linear_w8a16(const Tensor &input,
                    const Tensor &weight,
                    const Tensor &scale,
                    std::optional<Tensor> bias);

void linear_w8a16_(Tensor out,
                   const Tensor &input,
                   const Tensor &weight,
                   const Tensor &scale,
                   std::optional<Tensor> bias);

} // namespace infinicore::op
