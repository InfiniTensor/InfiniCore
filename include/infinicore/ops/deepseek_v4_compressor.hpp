#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"

#include <cstddef>

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(DeepseekV4Compressor,
                          Tensor,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          size_t,
                          float);

Tensor deepseek_v4_compressor(const Tensor &kv,
                              const Tensor &score,
                              const Tensor &ape,
                              const Tensor &norm_weight,
                              size_t compress_ratio,
                              float epsilon);

void deepseek_v4_compressor_(Tensor out,
                             const Tensor &kv,
                             const Tensor &score,
                             const Tensor &ape,
                             const Tensor &norm_weight,
                             size_t compress_ratio,
                             float epsilon);

} // namespace infinicore::op
