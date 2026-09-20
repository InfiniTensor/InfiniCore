#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"

#include <cstdint>

namespace infinicore::op {

// output = input @ dequant(weight)^T, with the weight left in its GGML block
// form: weight is a contiguous [N, row_bytes] U8 tensor and ggml_type is the
// enum ggml_type id of its blocks.  See infiniop/ops/linear_gguf.h for the
// accepted types and the batch limit.
INFINICORE_GRAPH_OP_CLASS(LinearGguf,
                          Tensor,
                          const Tensor &,
                          const Tensor &,
                          int64_t);

Tensor linear_gguf(const Tensor &input,
                   const Tensor &weight,
                   int64_t ggml_type);

void linear_gguf_(Tensor output,
                  const Tensor &input,
                  const Tensor &weight,
                  int64_t ggml_type);

} // namespace infinicore::op
