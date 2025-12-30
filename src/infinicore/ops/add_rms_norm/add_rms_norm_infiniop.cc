#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/add_rms_norm.hpp"
#include "infinicore/ops/common/cache.hpp"
#include <infiniop.h>

namespace infinicore::op::add_rms_norm_impl::infiniop {

thread_local common::OpCache<size_t, infiniopAddRMSNormDescriptor_t> caches(
    100, // capacity
    [](infiniopAddRMSNormDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyAddRMSNormDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor y, Tensor residual_out, Tensor a, Tensor b, Tensor weight, float epsilon) {
    size_t seed = hash_combine(y, residual_out, a, b, weight, epsilon);

    auto device = context::getDevice();
    auto &cache = caches.getCache(device);

    auto desc_opt = cache.get(seed);
    infiniopAddRMSNormDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateAddRMSNormDescriptor(
            context::getInfiniopHandle(device), &desc,
            y->desc(), a->desc(), b->desc(), weight->desc(), epsilon, residual_out->desc()));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetAddRMSNormWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopAddRMSNorm(
        desc, workspace->data(), workspace_size,
        y->data(), a->data(), b->data(), weight->data(), residual_out->data(), context::getStream()));
}

static bool registered = []() {
    AddRMSNorm::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::add_rms_norm_impl::infiniop
