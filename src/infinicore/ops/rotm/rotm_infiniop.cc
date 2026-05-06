#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/rotm.hpp"
#include <infiniop.h>

namespace infinicore::op::rotm_impl::infiniop {

thread_local common::OpCache<size_t, infiniopRotmDescriptor_t> caches(
    100,
    [](infiniopRotmDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyRotmDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor x, Tensor y, Tensor param) {
    size_t seed = hash_combine(x, y, param);

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();

    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopRotmDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateRotmDescriptor(
            context::getInfiniopHandle(x->device()), &desc,
            x->desc(), y->desc(), param->desc()));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetRotmWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopRotm(
        desc, workspace->data(), workspace_size,
        x->data(), y->data(), param->data(), context::getStream()));
}

static bool registered = []() {
    Rotm::dispatcher().registerDevice({Device::Type::CPU,
                                       Device::Type::CAMBRICON,
                                       Device::Type::METAX},
                                      &calculate,
                                      false);
    return true;
}();

} // namespace infinicore::op::rotm_impl::infiniop
