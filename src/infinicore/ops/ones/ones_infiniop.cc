#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/ones.hpp"
#include <infiniop.h>

namespace infinicore::op::ones_impl::infiniop {

thread_local common::OpCache<size_t, infiniopOnesDescriptor_t> caches(
    100,
    [](infiniopOnesDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyOnesDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor output) {
    size_t seed = 0;
    infinicore::hash_combine(seed, output);

    auto device = context::getDevice();
    auto &cache = caches.getCache(device);

    auto desc_opt = cache.get(seed);
    infiniopOnesDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateOnesDescriptor(
            context::getInfiniopHandle(device), &desc,
            output->desc(), output->desc()));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetOnesWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopOnes(
        desc, workspace->data(), workspace_size,
        output->data(), output->data(), context::getStream()));
}

static bool registered = []() {
    Ones::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::ones_impl::infiniop
