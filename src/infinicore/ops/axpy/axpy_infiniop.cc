#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/axpy.hpp"
#include "infinicore/ops/common/cache.hpp"
#include <infiniop.h>

namespace infinicore::op::axpy_impl::infiniop {

thread_local common::OpCache<size_t, infiniopAxpyDescriptor_t> caches(
    100, // capacity
    [](infiniopAxpyDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyAxpyDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor alpha, Tensor x, Tensor y) {
    size_t seed = hash_combine(alpha, x, y);

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();

    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopAxpyDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateAxpyDescriptor(
            context::getInfiniopHandle(y->device()), &desc,
            alpha->desc(), x->desc(), y->desc()));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetAxpyWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopAxpy(
        desc, workspace->data(), workspace_size,
        alpha->data(), x->data(), y->data(), context::getStream()));
}

static bool registered = []() {
    Axpy::dispatcher().registerDevice({Device::Type::CPU,
                                       Device::Type::CAMBRICON,
                                       Device::Type::METAX},
                                      &calculate,
                                      false);
    return true;
}();

} // namespace infinicore::op::axpy_impl::infiniop
