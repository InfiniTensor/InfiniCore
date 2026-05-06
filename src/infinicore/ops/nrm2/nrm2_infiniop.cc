#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/nrm2.hpp"
#include <infiniop.h>

namespace infinicore::op::nrm2_impl::infiniop {

thread_local common::OpCache<size_t, infiniopNrm2Descriptor_t> caches(
    100, // capacity
    [](infiniopNrm2Descriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyNrm2Descriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor result, Tensor x) {
    size_t seed = hash_combine(result, x);

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();

    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopNrm2Descriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateNrm2Descriptor(
            context::getInfiniopHandle(result->device()), &desc,
            x->desc(), result->desc()));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetNrm2WorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopNrm2(
        desc, workspace->data(), workspace_size,
        x->data(), result->data(), context::getStream()));
}

static bool registered = []() {
    Nrm2::dispatcher().registerDevice({Device::Type::CPU,
                                       Device::Type::CAMBRICON,
                                       Device::Type::METAX},
                                      &calculate,
                                      false);
    return true;
}();

} // namespace infinicore::op::nrm2_impl::infiniop
