#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/bi_attention.hpp"
#include "infinicore/ops/common/cache.hpp"
#include <infiniop.h>

namespace infinicore::op::bi_attention_impl::infiniop {

thread_local common::OpCache<size_t, infiniopBiAttentionDescriptor_t> caches(
    100, // capacity
    [](infiniopBiAttentionDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyBiAttentionDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor out, Tensor q, Tensor k, Tensor v, Tensor k_cache, Tensor v_cache, size_t pos) {
    size_t seed = hash_combine(out, q, k, v, k_cache, v_cache, pos);

    auto device = context::getDevice();
    auto &cache = caches.getCache(device);

    auto desc_opt = cache.get(seed);
    infiniopBiAttentionDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateBiAttentionDescriptor(
            context::getInfiniopHandle(device), &desc,
            out->desc(), q->desc(), k->desc(), v->desc(),
            k_cache->desc(), v_cache->desc(), pos));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetBiAttentionWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopBiAttention(
        desc, workspace->data(), workspace_size,
        out->data(), q->data(), k->data(), v->data(),
        k_cache->data(), v_cache->data(), context::getStream()));
}

static bool registered = []() {
    BiAttention::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::attention_impl::infiniop
