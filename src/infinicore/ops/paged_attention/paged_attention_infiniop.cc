#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/paged_attention.hpp"
#include <infiniop.h>

namespace infinicore::op::paged_attention_impl::infiniop {

thread_local common::OpCache<size_t, infiniopPagedAttentionDescriptor_t> caches(
    100, // capacity
    [](infiniopPagedAttentionDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyPagedAttentionDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor out, Tensor q, Tensor k_cache, Tensor v_cache, Tensor block_tables, Tensor seq_lens, std::optional<Tensor> alibi_slopes, float scale) {
    size_t seed = hash_combine(out, q, k_cache, v_cache, block_tables, seq_lens);

    auto device = context::getDevice();
    auto &cache = caches.getCache(device);

    auto desc_opt = cache.get(seed);
    infiniopPagedAttentionDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreatePagedAttentionDescriptor(
            context::getInfiniopHandle(device), &desc,
            out->desc(), q->desc(), k_cache->desc(), v_cache->desc(), block_tables->desc(), seq_lens->desc(),
            alibi_slopes.has_value() ? alibi_slopes.value()->desc() : nullptr,
            scale));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetPagedAttentionWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopPagedAttention(
        desc, workspace->data(), workspace_size,
        out->data(), q->data(), k_cache->data(), v_cache->data(), block_tables->data(), seq_lens->data(),
        alibi_slopes.has_value() ? alibi_slopes.value()->data() : nullptr,
        context::getStream()));
}

static bool registered = []() {
    PagedAttention::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::paged_attention_impl::infiniop
