#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(SwiGLUCuda, Tensor, const Tensor &, const Tensor &);

Tensor swiglu_cuda(const Tensor &a, const Tensor &b);
void swiglu_cuda_(Tensor c, const Tensor &a, const Tensor &b);

} // namespace infinicore::op
