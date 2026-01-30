#include "infinicore/ops/per_channel_quant_i8.hpp"
#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include <infiniop.h>
#include <iostream>

namespace infinicore::op::per_channel_quant_i8_impl::infiniop {

thread_local common::OpCache<size_t, infiniopPerChannelQuantI8Descriptor_t> caches(
    100, // capacity
    [](infiniopPerChannelQuantI8Descriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyPerChannelQuantI8Descriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor x, Tensor x_packed, Tensor x_scale) {
    size_t seed = hash_combine(x, x_packed, x_scale);

    auto device = context::getDevice();
    auto &cache = caches.getCache(device);

    auto desc_opt = cache.get(seed);
    infiniopGemmDescriptor_t desc = nullptr;
    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreatePerChannelQuantI8Descriptor(
            context::getInfiniopHandle(device), &desc,
            x_packed->desc(), x_scale->desc(), nullptr, x->desc()));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetPerChannelQuantI8WorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);
    INFINICORE_CHECK_ERROR(infiniopPerChannelQuantI8(
        desc, workspace->data(), workspace_size,
        x_packed->data(), x_scale->data(), nullptr, x->data(), context::getStream()));
}

static bool registered = []() {
    PerChannelQuantI8::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::per_channel_quant_i8_impl::infiniop
