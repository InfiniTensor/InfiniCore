#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/cosh.hpp"
#include <infiniop.h>

namespace infinicore::op::cosh_impl::infiniop {

thread_local common::OpCache<size_t, infiniopCoshDescriptor_t> caches(
    100,
    [](infiniopCoshDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyCoshDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor y, Tensor x) {
    size_t seed = hash_combine(y, x);

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();

    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopCoshDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateCoshDescriptor(
            context::getInfiniopHandle(y->device()), &desc,
            y->desc(), x->desc()));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetCoshWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopCosh(
        desc, workspace->data(), workspace_size,
        y->data(), x->data(), context::getStream()));
}

static bool registered = []() {
    Cosh::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::cosh_impl::infiniop