#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/rotmg.hpp"
#include <infiniop.h>

namespace infinicore::op::rotmg_impl::infiniop {

thread_local common::OpCache<size_t, infiniopRotmgDescriptor_t> caches(
    100,
    [](infiniopRotmgDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyRotmgDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor d1, Tensor d2, Tensor x1, Tensor y1, Tensor param) {
    size_t seed = hash_combine(d1, d2, x1, y1, param);

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();

    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopRotmgDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateRotmgDescriptor(
            context::getInfiniopHandle(d1->device()), &desc,
            d1->desc(), d2->desc(), x1->desc(), y1->desc(), param->desc()));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetRotmgWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopRotmg(
        desc, workspace->data(), workspace_size,
        d1->data(), d2->data(), x1->data(), y1->data(), param->data(), context::getStream()));
}

static bool registered = []() {
    Rotmg::dispatcher().registerDevice({Device::Type::CPU,
                                        Device::Type::CAMBRICON,
                                        Device::Type::METAX},
                                       &calculate,
                                       false);
    return true;
}();

} // namespace infinicore::op::rotmg_impl::infiniop
