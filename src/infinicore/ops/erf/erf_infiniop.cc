#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/erf.hpp"
#include <infiniop.h>

namespace infinicore::op::erf_impl::infiniop {

thread_local common::OpCache<size_t, infiniopErfDescriptor_t> caches(
    100, // capacity
    [](infiniopErfDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyErfDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor output, Tensor input) {
    size_t seed = hash_combine(output, input);

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();

    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopErfDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateErfDescriptor(
            context::getInfiniopHandle(output->device()), &desc,
            output->desc(), input->desc()));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetErfWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopErf(
        desc, workspace->data(), workspace_size,
        output->data(), input->data(), context::getStream()));
}

static bool registered = []() {
    Erf::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::erf_impl::infiniop

