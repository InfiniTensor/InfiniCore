#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(Ernie45MRoPE, Tensor, Tensor, const Tensor &, double, size_t, size_t, size_t);
INFINICORE_GRAPH_OP_CLASS(Ernie45VisionRoPE, Tensor, Tensor, const Tensor &, double);

Tensor ernie45_mrope_(Tensor q,
                      Tensor k,
                      const Tensor &positions,
                      double rope_theta,
                      size_t section_h,
                      size_t section_w,
                      size_t section_t);

Tensor ernie45_vision_rope_(Tensor q,
                            Tensor k,
                            const Tensor &positions,
                            double rope_theta);

} // namespace infinicore::op
