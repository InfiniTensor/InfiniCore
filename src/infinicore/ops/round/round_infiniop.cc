#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/round.hpp"
#include <infiniop.h>

namespace infinicore::op::round_impl::infiniop {

thread_local common::OpCache<size_t, infiniopRoundDescriptor_t> caches(
    100,
    [](infiniopRoundDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyRoundDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor y, Tensor x, int decimals) {
    size_t seed = hash_combine(y, x, decimals);

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();

    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopRoundDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateRoundDescriptor(
            context::getInfiniopHandle(y->device()), &desc,
            y->desc(), x->desc(), decimals));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetRoundWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    // Guard against cross-stream producer/consumer ordering issues in callers.
    context::syncDevice();

    INFINICORE_CHECK_ERROR(infiniopRound(
        desc, workspace->data(), workspace_size,
        y->data(), x->data(), context::getStream()));

    // Ensure temporary workspace is not released before the CUDA work completes.
    context::syncStream();
}

static bool registered = []() {
    Round::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::round_impl::infiniop
