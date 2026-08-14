#pragma once

#include "../tensor.hpp"

#include <cstddef>

namespace infinicore::op {

enum class HygonMoeMarlinWeightFormat {
    W16A16,
    W8A8,
};

struct HygonMoeMarlinWeights {
    Tensor packed_w13;
    Tensor packed_w2;
    Tensor packed_w13_scale;
    Tensor packed_w2_scale;
    HygonMoeMarlinWeightFormat format = HygonMoeMarlinWeightFormat::W16A16;
};

struct HygonMoeMarlinWorkspace {
    Tensor output;
    Tensor cache13;
    Tensor cache2;
    Tensor input_i8;
    Tensor input_scale;
    Tensor cache2_i8;
    Tensor cache2_scale;
    Tensor sorted_token_ids;
    Tensor expert_ids;
    Tensor num_tokens_post_padded;

    size_t cache13_capacity = 0;
    size_t cache2_capacity = 0;
    size_t sorted_token_ids_capacity = 0;
    size_t expert_ids_capacity = 0;
};

struct HygonMoeMarlinOutput {
    Tensor hidden_states;
    Tensor sorted_token_ids;
    Tensor expert_ids;
    Tensor num_tokens_post_padded;
    bool has_routing_metadata = false;
};

HygonMoeMarlinOutput hygon_moe_marlin_fused(
    const Tensor &hidden_states,
    const Tensor &topk_weights,
    const Tensor &topk_ids,
    const Tensor &expert_map,
    const HygonMoeMarlinWeights &weights,
    HygonMoeMarlinWorkspace &workspace,
    size_t num_local_experts,
    size_t hidden_size,
    size_t intermediate_size,
    size_t fallback_align_block_size);

} // namespace infinicore::op
