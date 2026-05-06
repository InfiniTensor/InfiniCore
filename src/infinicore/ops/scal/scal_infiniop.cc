#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/scal.hpp"
#include <infiniop.h>

namespace infinicore::op::scal_impl::infiniop {

thread_local common::OpCache<size_t, infiniopScalDescriptor_t> caches(
    100, // capacity
    [](infiniopScalDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyScalDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor alpha, Tensor x) {
    size_t seed = hash_combine(alpha, x);

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();

    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopScalDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateScalDescriptor(
            context::getInfiniopHandle(x->device()), &desc,
            alpha->desc(), x->desc()));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetScalWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopScal(
        desc, workspace->data(), workspace_size,
        alpha->data(), x->data(), context::getStream()));
}

static bool registered = []() {
    Scal::dispatcher().registerDevice({Device::Type::CPU,
                                       Device::Type::CAMBRICON,
                                       Device::Type::METAX},
                                      &calculate,
                                      false);
    return true;
}();

} // namespace infinicore::op::scal_impl::infiniop
