#pragma once

#include "../device.hpp"
#include "../graph/graph.hpp"
#include "../tensor.hpp"
#include "common/op.hpp"

#include <tuple>

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(DeepseekV4TopkRouter,
                          Tensor,
                          Tensor,
                          const Tensor &,
                          const Tensor &,
                          bool);

INFINICORE_GRAPH_OP_CLASS(DeepseekV4HashRouter,
                          Tensor,
                          Tensor,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          bool);

INFINICORE_GRAPH_OP_CLASS(DeepseekV4HashTopkRouter,
                          Tensor,
                          Tensor,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          const Tensor &,
                          bool);

std::tuple<Tensor, Tensor> deepseek_v4_topk_router(
    const Tensor &logits,
    size_t topk,
    bool renormalize,
    const Tensor &bias = Tensor());

void deepseek_v4_topk_router_(
    Tensor topk_weights,
    Tensor topk_indices,
    const Tensor &logits,
    const Tensor &bias = Tensor(),
    bool renormalize = true);

std::tuple<Tensor, Tensor> deepseek_v4_hash_router(
    const Tensor &logits,
    const Tensor &input_ids,
    const Tensor &tid2eid,
    bool renormalize);

void deepseek_v4_hash_router_(
    Tensor topk_weights,
    Tensor topk_indices,
    const Tensor &logits,
    const Tensor &input_ids,
    const Tensor &tid2eid,
    bool renormalize = true);

std::tuple<Tensor, Tensor> deepseek_v4_hash_topk_router(
    const Tensor &hidden_states,
    const Tensor &weight,
    const Tensor &input_ids,
    const Tensor &tid2eid,
    bool renormalize);

void deepseek_v4_hash_topk_router_(
    Tensor topk_weights,
    Tensor topk_indices,
    const Tensor &hidden_states,
    const Tensor &weight,
    const Tensor &input_ids,
    const Tensor &tid2eid,
    bool renormalize = true);

} // namespace infinicore::op
