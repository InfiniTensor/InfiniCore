#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/paged_attention_prefill.hpp"
#include <infiniop.h>

namespace infinicore::op::paged_attention_prefill_impl::infiniop {

thread_local common::OpCache<size_t, infiniopPagedAttentionPrefillDescriptor_t> caches(
    100, // capacity
    [](infiniopPagedAttentionPrefillDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyPagedAttentionPrefillDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor out, Tensor q, Tensor k_cache, Tensor v_cache, Tensor block_tables, Tensor cache_lens, Tensor seq_lens, Tensor seq_offsets, std::optional<Tensor> alibi_slopes, float scale) {
    size_t seed = hash_combine(out, q, k_cache, v_cache, block_tables, cache_lens, seq_lens, seq_offsets, alibi_slopes, scale);

    auto device = context::getDevice();
    auto &cache = caches.getCache(device);

    auto desc_opt = cache.get(seed);
    infiniopPagedAttentionPrefillDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreatePagedAttentionPrefillDescriptor(
            context::getInfiniopHandle(device), &desc,
            out->desc(), q->desc(), k_cache->desc(), v_cache->desc(), block_tables->desc(),
            cache_lens->desc(), seq_lens->desc(), seq_offsets->desc(),
            alibi_slopes.has_value() ? alibi_slopes.value()->desc() : nullptr,
            scale));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetPagedAttentionPrefillWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopPagedAttentionPrefill(
        desc, workspace->data(), workspace_size,
        out->data(), q->data(), k_cache->data(), v_cache->data(), block_tables->data(), cache_lens->data(), seq_lens->data(), seq_offsets->data(),
        alibi_slopes.has_value() ? alibi_slopes.value()->data() : nullptr,
        context::getStream()));
}

static bool registered = []() {
    PagedAttentionPrefill::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::paged_attention_prefill_impl::infiniop
