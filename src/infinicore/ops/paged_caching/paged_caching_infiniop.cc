#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/paged_caching.hpp"
#include <infiniop.h>

namespace infinicore::op::paged_caching_impl::infiniop {

thread_local common::OpCache<size_t, infiniopPagedCachingDescriptor_t> caches(
    100, // capacity
    [](infiniopPagedCachingDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyPagedCachingDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor k_cache, Tensor v_cache, Tensor k, Tensor v, Tensor slot_mapping) {
    size_t seed = hash_combine(k_cache, v_cache, k, v, slot_mapping);

    auto device = context::getDevice();
    auto &cache = caches.getCache(device);

    auto desc_opt = cache.get(seed);
    infiniopPagedCachingDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreatePagedCachingDescriptor(
            context::getInfiniopHandle(device), &desc,
            k_cache->desc(), v_cache->desc(), k->desc(), v->desc(), slot_mapping->desc()));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetPagedCachingWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopPagedCaching(
        desc, workspace->data(), workspace_size,
        k_cache->data(), v_cache->data(), k->data(), v->data(), slot_mapping->data(), context::getStream()));
}

static bool registered = []() {
    PagedCaching::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::paged_caching_impl::infiniop
