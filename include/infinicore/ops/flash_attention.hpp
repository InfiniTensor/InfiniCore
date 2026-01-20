#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_CLASS(FlashAttention, Tensor, Tensor, Tensor, Tensor, std::size_t, float, bool);

Tensor flash_attention(Tensor q, Tensor k, Tensor v, std::size_t total_kv_len, float scale, bool is_causal);
void flash_attention_(Tensor out, Tensor q, Tensor k, Tensor v, std::size_t total_kv_len, float scale, bool is_causal);
} // namespace infinicore::op
