#pragma once
#include "../graph/graph.hpp"
#include "common/op.hpp"

namespace infinicore::op {
INFINICORE_GRAPH_OP_CLASS(Mamba2Scan, Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, Tensor, const Tensor &, const Tensor &, const Tensor &);

// Packed Mamba-2 scan. State row zero is read-only; final rows must be unique.
__export Tensor mamba2_scan(const Tensor &x, const Tensor &dt, const Tensor &b, const Tensor &c, const Tensor &a, const Tensor &d, const Tensor &dt_bias, Tensor state, const Tensor &offsets, const Tensor &initial_indices, const Tensor &final_indices);
__export void mamba2_scan_(Tensor out, const Tensor &x, const Tensor &dt, const Tensor &b, const Tensor &c, const Tensor &a, const Tensor &d, const Tensor &dt_bias, Tensor state, const Tensor &offsets, const Tensor &initial_indices, const Tensor &final_indices);
} // namespace infinicore::op
