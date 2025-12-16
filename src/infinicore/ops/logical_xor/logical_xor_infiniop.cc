#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/logical_xor.hpp"
#include <infiniop.h>

namespace infinicore::op::logical_xor_impl::infiniop {

thread_local common::OpCache<size_t, infiniopLogicalXorDescriptor_t> caches(
    100,
    [](infiniopLogicalXorDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyLogicalXorDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor c, Tensor a, Tensor b) {
    size_t seed = hash_combine(c, a, b);

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();

    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopLogicalXorDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateLogicalXorDescriptor(
            context::getInfiniopHandle(c->device()), &desc,
            c->desc(), a->desc(), b->desc()));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetLogicalXorWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopLogicalXor(
        desc, workspace->data(), workspace_size,
        c->data(), a->data(), b->data(), context::getStream()));
}

static bool registered = []() {
    LogicalXor::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::logical_xor_impl::infiniop
