#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/rearrange.hpp"
#include <infiniop.h>

namespace infinicore::op::rearrange_impl::infiniop {

thread_local common::OpCache<size_t, infiniopRearrangeDescriptor_t> caches(
    100, // capacity
    [](infiniopRearrangeDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyRearrangeDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor y, Tensor x) {
    size_t seed = hash_combine(y, x);

    auto device_type = y->device().getType();
    auto device_index = y->device().getIndex();

    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopRearrangeDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateRearrangeDescriptor(context::getInfiniopHandle(y->device()), &desc, y->desc(), x->desc()));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    INFINICORE_CHECK_ERROR(
        infiniopRearrange(
            desc,
            y->data(),
            x->data(),
            context::getStream()));
}

static bool registered = []() {
    Rearrange::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::rearrange_impl::infiniop
