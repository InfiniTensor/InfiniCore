#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/rms_norm.hpp"
#include <infiniop.h>

namespace infinicore::op::rms_norm_impl::infiniop {

thread_local common::OpCache<size_t, infiniopRMSNormDescriptor_t> caches(
    100, // capacity
    [](infiniopRMSNormDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyRMSNormDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor y, Tensor x, Tensor weight, float epsilon) {
    size_t seed = hash_combine(y, x, weight, epsilon);

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();

    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopRMSNormDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateRMSNormDescriptor(
            context::getInfiniopHandle(), &desc,
            y->desc(), x->desc(), weight->desc(), epsilon));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetRMSNormWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopRMSNorm(
        desc, workspace->data(), workspace_size,
        y->data(), x->data(), weight->data(), context::getStream()));
}

static bool registered = []() {
    RMSNorm::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::rms_norm_impl::infiniop
