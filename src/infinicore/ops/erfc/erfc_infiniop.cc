#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/erfc.hpp"
#include <infiniop.h>

namespace infinicore::op::erfc_impl::infiniop {

thread_local common::OpCache<size_t, infiniopErfcDescriptor_t> caches(
    100, // capacity
    [](infiniopErfcDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyErfcDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor output, Tensor input) {
    size_t seed = hash_combine(output, input);

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();

    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopErfcDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateErfcDescriptor(
            context::getInfiniopHandle(output->device()), &desc,
            output->desc(), input->desc()));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetErfcWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopErfc(
        desc, workspace->data(), workspace_size,
        output->data(), input->data(), context::getStream()));
}

static bool registered = []() {
    Erfc::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::erfc_impl::infiniop

