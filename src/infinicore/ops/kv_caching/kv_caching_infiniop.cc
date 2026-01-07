#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/kv_caching.hpp"
#include <infiniop.h>

namespace infinicore::op::kv_caching_impl::infiniop {

thread_local common::OpCache<size_t, infiniopKVCachingDescriptor_t> caches(
    100, // capacity
    [](infiniopKVCachingDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyKVCachingDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor k_cache,
               Tensor v_cache,
               Tensor k,
               Tensor v,
               Tensor offsets,
               Tensor past_kv_lengths,
               Tensor cache_ids) {
    size_t seed = hash_combine(k_cache, v_cache, k, v, offsets, past_kv_lengths, cache_ids);

    auto device = context::getDevice();
    auto &cache = caches.getCache(device);

    auto desc_opt = cache.get(seed);
    infiniopKVCachingDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateKVCachingDescriptor(
            context::getInfiniopHandle(device), &desc,
            k_cache->desc(), v_cache->desc(),
            k->desc(), v->desc(),
            offsets->desc(), past_kv_lengths->desc(),
            cache_ids->desc()));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetKVCachingWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopKVCaching(
        desc, workspace->data(), workspace_size,
        k_cache->data(), v_cache->data(),
        k->data(), v->data(),
        offsets->data(), past_kv_lengths->data(), cache_ids->data(),
        context::getStream()));
}

static bool registered = []() {
    KVCaching::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::kv_caching_impl::infiniop
