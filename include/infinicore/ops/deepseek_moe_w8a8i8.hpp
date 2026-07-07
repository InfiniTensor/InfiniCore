#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "common/op.hpp"
#include <vector>

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(
    DeepseekMoeW8A8I8,
    Tensor,
    const Tensor &,
    const Tensor &,
    const Tensor &,
    const std::vector<Tensor> &,
    const std::vector<Tensor> &,
    const std::vector<Tensor> &,
    const std::vector<Tensor> &,
    const std::vector<Tensor> &,
    const std::vector<Tensor> &,
    size_t,
    size_t);

Tensor deepseek_moe_w8a8i8(const Tensor &hidden,
                           const Tensor &topk_indices,
                           const Tensor &topk_weights,
                           const std::vector<Tensor> &gate_weights,
                           const std::vector<Tensor> &up_weights,
                           const std::vector<Tensor> &down_weights,
                           const std::vector<Tensor> &gate_weight_scales,
                           const std::vector<Tensor> &up_weight_scales,
                           const std::vector<Tensor> &down_weight_scales,
                           size_t intermediate_size,
                           size_t num_experts);

void deepseek_moe_w8a8i8_(Tensor out,
                          const Tensor &hidden,
                          const Tensor &topk_indices,
                          const Tensor &topk_weights,
                          const std::vector<Tensor> &gate_weights,
                          const std::vector<Tensor> &up_weights,
                          const std::vector<Tensor> &down_weights,
                          const std::vector<Tensor> &gate_weight_scales,
                          const std::vector<Tensor> &up_weight_scales,
                          const std::vector<Tensor> &down_weight_scales,
                          size_t intermediate_size,
                          size_t num_experts);

} // namespace infinicore::op
