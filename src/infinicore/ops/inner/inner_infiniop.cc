#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/inner.hpp"
#include "infinicore/ops/common/cache.hpp"
#include <infiniop.h>

namespace infinicore::op::inner_impl::infiniop {

thread_local common::OpCache<size_t, infiniopInnerDescriptor_t> caches(
    100,
    [](infiniopInnerDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyInnerDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor input, Tensor other, Tensor out) {
    size_t seed = hash_combine(input, other, out);

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();

    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopInnerDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateInnerDescriptor(
            context::getInfiniopHandle(input->device()), &desc,
            input->desc(), other->desc(), out->desc()));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetInnerWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);


    INFINICORE_CHECK_ERROR(infiniopInner(
        desc, workspace->data(), workspace_size,
        input->data(), other->data(), out->data(), context::getStream()));
}

static bool registered = []() {
    Inner::dispatcher().registerDevice({
            Device::Type::CPU,
            Device::Type::NVIDIA
            // Device::Type::METAX,
            // Device::Type::MOORE,
            // Device::Type::ILUVATAR
        }, &calculate, false);
    return true;
}();

}