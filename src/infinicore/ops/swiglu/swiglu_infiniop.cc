#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/swiglu.hpp"
#include <infiniop.h>

namespace infinicore::op::swiglu_impl::infiniop {

thread_local common::OpCache<size_t, infiniopSwiGLUDescriptor_t> caches(
    100, // capacity
    [](infiniopSwiGLUDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroySwiGLUDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor c, Tensor a, Tensor b) {
    size_t seed = hash_combine(c, b, a);

    auto device = context::getDevice();
    auto &cache = caches.getCache(device);

    auto desc_opt = cache.get(seed);
    infiniopSwiGLUDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateSwiGLUDescriptor(
            context::getInfiniopHandle(device), &desc,
            c->desc(), a->desc(), b->desc()));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetSwiGLUWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopSwiGLU(
        desc, workspace->data(), workspace_size,
        c->data(), a->data(), b->data(), context::getStream()));
}

static bool registered = []() {
    SwiGLU::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::swiglu_impl::infiniop
