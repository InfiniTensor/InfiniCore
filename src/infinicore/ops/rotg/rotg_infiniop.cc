#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/rotg.hpp"
#include <infiniop.h>

namespace infinicore::op::rotg_impl::infiniop {

thread_local common::OpCache<size_t, infiniopRotgDescriptor_t> caches(
    100,
    [](infiniopRotgDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyRotgDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor x, Tensor y, Tensor c, Tensor s) {
    size_t seed = hash_combine(x, y, c, s);

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();

    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopRotgDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateRotgDescriptor(
            context::getInfiniopHandle(x->device()), &desc,
            x->desc(), y->desc(), c->desc(), s->desc()));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetRotgWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopRotg(
        desc, workspace->data(), workspace_size,
        x->data(), y->data(), c->data(), s->data(), context::getStream()));
}

static bool registered = []() {
    Rotg::dispatcher().registerDevice({Device::Type::CPU,
                                       Device::Type::CAMBRICON,
                                       Device::Type::METAX},
                                      &calculate,
                                      false);
    return true;
}();

} // namespace infinicore::op::rotg_impl::infiniop
