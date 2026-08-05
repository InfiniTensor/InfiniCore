#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"

#include <optional>

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(LinearMxfp4,
                          Tensor,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          std::optional<Tensor>,
                          float);

Tensor linear_mxfp4(const Tensor &input,
                    const Tensor &packed_weight,
                    const Tensor &weight_scale,
                    std::optional<Tensor> bias = std::nullopt,
                    float alpha = 1.0f);

void linear_mxfp4_(Tensor output,
                   const Tensor &input,
                   const Tensor &packed_weight,
                   const Tensor &weight_scale,
                   std::optional<Tensor> bias = std::nullopt,
                   float alpha = 1.0f);

} // namespace infinicore::op
