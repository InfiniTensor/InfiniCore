#include "infinicore/ops/paged_attention.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

common::OpDispatcher<PagedAttention::schema> &PagedAttention::dispatcher() {
    static common::OpDispatcher<PagedAttention::schema> dispatcher_;
    return dispatcher_;
};

void PagedAttention::execute(Tensor out, Tensor q, Tensor k_cache, Tensor v_cache, Tensor block_tables, Tensor cache_lens, std::optional<Tensor> alibi_slopes, float scale) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, q, k_cache, v_cache, block_tables, cache_lens);
    infinicore::context::setDevice(out->device());
    dispatcher().lookup(out->device().getType())(out, q, k_cache, v_cache, block_tables, cache_lens, alibi_slopes, scale);
}

Tensor paged_attention(Tensor q, Tensor k_cache, Tensor v_cache, Tensor block_tables, Tensor cache_lens, std::optional<Tensor> alibi_slopes, float scale) {
    auto out = Tensor::empty(q->shape(), q->dtype(), q->device());
    paged_attention_(out, q, k_cache, v_cache, block_tables, cache_lens, alibi_slopes, scale);
    return out;
}

void paged_attention_(Tensor out, Tensor q, Tensor k_cache, Tensor v_cache, Tensor block_tables, Tensor cache_lens, std::optional<Tensor> alibi_slopes, float scale) {
    PagedAttention::execute(out, q, k_cache, v_cache, block_tables, cache_lens, alibi_slopes, scale);
}

} // namespace infinicore::op
