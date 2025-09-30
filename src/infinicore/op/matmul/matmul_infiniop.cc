#include "infinicore/common/utils.hpp"
#include "infinicore/op/common/cache.hpp"
#include "infinicore/op/matmul.hpp"
#include <infiniop.h>

namespace infinicore::op::matmul_impl::infiniop {

thread_local common::OpCache<size_t, infiniopGemmDescriptor_t> caches(
    100, // capacity
    [](infiniopGemmDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyGemmDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor c, Tensor a, Tensor b) {
    size_t seed = hash_combine(c, b, a);

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();

    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopGemmDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateGemmDescriptor(context::getInfiniopHandle(), &desc, c->desc(), a->desc(), b->desc()));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetGemmWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopGemm(
        desc, workspace->data(), workspace_size,
        c->data(), a->data(), b->data(), 1.f, 0.f, context::getStream()));
}

static bool registered = []() {
    Matmul::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::matmul_impl::infiniop
