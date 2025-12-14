#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/pixel_shuffle.hpp"
#include <infiniop.h>

namespace infinicore::op::pixel_shuffle_impl::infiniop {

thread_local common::OpCache<size_t, infiniopPixelShuffleDescriptor_t> caches(
    100, // capacity
    [](infiniopPixelShuffleDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyPixelShuffleDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor output, Tensor input, int64_t upscale_factor) {
    size_t seed = hash_combine(output, input, upscale_factor);

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();

    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopPixelShuffleDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreatePixelShuffleDescriptor(
            context::getInfiniopHandle(output->device()), &desc,
            output->desc(), input->desc(), static_cast<int>(upscale_factor)));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetPixelShuffleWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopPixelShuffle(
        desc, workspace->data(), workspace_size,
        output->data(), input->data(), context::getStream()));
}

static bool registered = []() {
    PixelShuffle::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::pixel_shuffle_impl::infiniop

