#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/vdot.hpp"
#include <infiniop.h>

namespace infinicore::op::vdot_impl::infiniop {

thread_local common::OpCache<size_t, infiniopVdotDescriptor_t> caches(
    100,
    [](infiniopVdotDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyVdotDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor out, Tensor a, Tensor b) {
    size_t seed = hash_combine(out, a, b);

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();

    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopVdotDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateVdotDescriptor(
            context::getInfiniopHandle(out->device()), &desc,
            out->desc(), a->desc(), b->desc()));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetVdotWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopVdot(
        desc, workspace->data(), workspace_size,
        out->data(), a->data(), b->data(), context::getStream()));
}

static bool registered = []() {
    Vdot::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::vdot_impl::infiniop


