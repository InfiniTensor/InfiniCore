#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/blas_amin.hpp"
#include "infinicore/ops/common/cache.hpp"
#include <infiniop.h>

namespace infinicore::op::blas_amin_impl::infiniop {

thread_local common::OpCache<size_t, infiniopBlasAminDescriptor_t> caches(
    100, // capacity
    [](infiniopBlasAminDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyBlasAminDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor result, Tensor x) {
    size_t seed = hash_combine(result, x);

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();

    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopBlasAminDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateBlasAminDescriptor(
            context::getInfiniopHandle(result->device()), &desc,
            x->desc(), result->desc()));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetBlasAminWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopBlasAmin(
        desc, workspace->data(), workspace_size,
        x->data(), result->data(), context::getStream()));
}

static bool registered = []() {
    BlasAmin::dispatcher().registerDevice({Device::Type::CPU,
                                           Device::Type::CAMBRICON,
                                           Device::Type::METAX},
                                          &calculate,
                                          false);
    return true;
}();

} // namespace infinicore::op::blas_amin_impl::infiniop
