#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(Dsv4SiluMulMaskedQuant,
                          Tensor,
                          Tensor,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &);

void dsv4_silu_mul_masked_quant_(Tensor q, Tensor scale, const Tensor &gate, const Tensor &up, const Tensor &mask);

} // namespace infinicore::op
