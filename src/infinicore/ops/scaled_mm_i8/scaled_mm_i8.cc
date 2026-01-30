#include "infinicore/ops/scaled_mm_i8.hpp"
#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include <infiniop.h>

namespace infinicore::op::scaled_mm_i8_impl::infiniop {

thread_local common::OpCache<size_t, infiniopI8GemmDescriptor_t> caches(
    100, // capacity
    [](infiniopI8GemmDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyI8GemmDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor c, Tensor a_p, Tensor a_s, Tensor b_p, Tensor b_s, std::optional<Tensor> bias) {
    size_t seed = hash_combine(c, a_p, a_s, b_p, b_s);

    auto device = context::getDevice();
    auto &cache = caches.getCache(device);

    auto desc_opt = cache.get(seed);
    infiniopGemmDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateI8GemmDescriptor(
            context::getInfiniopHandle(device), &desc,
            c->desc(), bias.has_value() ? bias.value()->desc() : nullptr, a_p->desc(), a_s->desc(), b_p->desc(), b_s->desc()));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetI8GemmWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopI8Gemm(
        desc, workspace->data(), workspace_size,
        c->data(), bias.has_value() ? bias.value()->data() : nullptr, a_p->data(), a_s->data(), b_p->data(), b_s->data(), context::getStream()));
}

static bool registered = []() {
    ScaledMMI8::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::scaled_mm_i8_impl::infiniop
