#pragma once

#include "../device.hpp"
#include "common/op.hpp"
#include <optional>

namespace infinicore::op {

class PagedAttentionPrefill {
public:
    using schema = void (*)(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, std::optional<Tensor>, float);
    static void execute(Tensor out, Tensor q, Tensor k_cache, Tensor v_cache, Tensor block_tables, Tensor cache_lens, Tensor seq_lens, Tensor seq_offsets, std::optional<Tensor> alibi_slopes, float);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor paged_attention_prefill(Tensor q, Tensor k_cache, Tensor v_cache, Tensor block_tables, Tensor cache_lens, Tensor seq_lens, Tensor seq_offsets, std::optional<Tensor> alibi_slopes, float scale);
void paged_attention_prefill_(Tensor out, Tensor q, Tensor k_cache, Tensor v_cache, Tensor block_tables, Tensor cache_lens, Tensor seq_lens, Tensor seq_offsets, std::optional<Tensor> alibi_slopes, float scale);
} // namespace infinicore::op
