#pragma once

#include "../device.hpp"
#include "common/op.hpp"
#include <optional>

namespace infinicore::op {

class PagedAttention {
public:
    using schema = void (*)(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, std::optional<Tensor>, float);
    static void execute(Tensor out, Tensor q, Tensor k_cache, Tensor v_cache, Tensor block_tables, Tensor cache_lens, std::optional<Tensor> alibi_slopes, float);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor paged_attention(Tensor q, Tensor k_cache, Tensor v_cache, Tensor block_tables, Tensor cache_lens, std::optional<Tensor> alibi_slopes, float scale);
void paged_attention_(Tensor out, Tensor q, Tensor k_cache, Tensor v_cache, Tensor block_tables, Tensor cache_lens, std::optional<Tensor> alibi_slopes, float scale);
} // namespace infinicore::op
