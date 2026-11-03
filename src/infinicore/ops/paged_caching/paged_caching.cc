#include "infinicore/ops/paged_caching.hpp"
#include "../../utils.hpp"

namespace infinicore::op::paged_caching_impl::infiniop {
bool needs_slot_validation(void *);
void validate_slots_before_replay(void *, graph::GraphReplayStage);
}

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(PagedCaching);

PagedCaching::PagedCaching(Tensor k_cache, Tensor v_cache, const Tensor &k, const Tensor &v, const Tensor &slot_mapping) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(k_cache, v_cache, k, v, slot_mapping);
    INFINICORE_GRAPH_OP_DISPATCH(k->device().getType(), k_cache, v_cache, k, v, slot_mapping);
    if (k->device().getType() == Device::Type::ASCEND
        && paged_caching_impl::infiniop::needs_slot_validation(planned_meta_)) {
        // PREPARE_UPDATE validates before an asynchronous launch. UPDATE also
        // covers graphs whose other operators require launch-after-update.
        set_graph_replay_handler(&paged_caching_impl::infiniop::validate_slots_before_replay, true);
    }
}

void PagedCaching::execute(Tensor k_cache, Tensor v_cache, const Tensor &k, const Tensor &v, const Tensor &slot_mapping) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(PagedCaching, k_cache, v_cache, k, v, slot_mapping);
}

void paged_caching_(Tensor k_cache, Tensor v_cache, const Tensor &k, const Tensor &v, const Tensor &slot_mapping) {
    constexpr Size MAX_TOKENS_PER_LAUNCH = 32768;
    const Size num_tokens = k->size(0);

    for (Size start = 0; start < num_tokens; start += MAX_TOKENS_PER_LAUNCH) {
        const Size remaining = num_tokens - start;
        const Size chunk_size = remaining < MAX_TOKENS_PER_LAUNCH ? remaining : MAX_TOKENS_PER_LAUNCH;
        PagedCaching::execute(k_cache, v_cache,
                              k->narrow({{0, start, chunk_size}}),
                              v->narrow({{0, start, chunk_size}}),
                              slot_mapping->narrow({{0, start, chunk_size}}));
    }
}

} // namespace infinicore::op
