#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"

#include <cstddef>
#include <cstdint>

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(DeepseekV4SwaDecode,
                          Tensor,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          float,
                          size_t,
                          double,
                          bool,
                          double,
                          double,
                          double,
                          int64_t,
                          double);

Tensor deepseek_v4_swa_decode(const Tensor &q,
                              const Tensor &k,
                              const Tensor &attn_sink,
                              const Tensor &positions,
                              float softmax_scale,
                              size_t rope_dim,
                              double rope_theta,
                              bool use_yarn,
                              double yarn_factor,
                              double yarn_beta_fast,
                              double yarn_beta_slow,
                              int64_t yarn_original_seq_len,
                              double yarn_extrapolation_factor);

void deepseek_v4_swa_decode_(Tensor y,
                             const Tensor &q,
                             const Tensor &k,
                             const Tensor &attn_sink,
                             const Tensor &positions,
                             float softmax_scale,
                             size_t rope_dim,
                             double rope_theta,
                             bool use_yarn,
                             double yarn_factor,
                             double yarn_beta_fast,
                             double yarn_beta_slow,
                             int64_t yarn_original_seq_len,
                             double yarn_extrapolation_factor);

} // namespace infinicore::op
