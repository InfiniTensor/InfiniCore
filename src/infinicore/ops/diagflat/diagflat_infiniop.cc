#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/diagflat.hpp"
#include "infiniop/ops/diagflat.h"
#include <infiniop.h>



namespace infinicore::op::diagflat_impl::infiniop {

thread_local common::OpCache<size_t, infiniopDiagflatDescriptor_t> caches(
    100,
    [](infiniopDiagflatDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyDiagflatDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor output, Tensor input, int64_t offset) {
    size_t seed = 0;
    hash_combine(seed, static_cast<size_t>(input->dtype()));
    for (Size s : input->shape()) {
        hash_combine(seed, static_cast<size_t>(s));
    }
    for (Stride st : input->strides()) {
        hash_combine(seed, static_cast<size_t>(st));
    }
    hash_combine(seed, static_cast<size_t>(offset));

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();
    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopDiagflatDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateDiagflatDescriptor(
            context::getInfiniopHandle(output->device()),
            &desc,
            output->desc(),
            input->desc(),
            offset));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetDiagflatWorkspaceSize(desc, &workspace_size));

    std::shared_ptr<Memory> workspace;
    void *workspace_ptr = nullptr;
    if (workspace_size > 0) {
        workspace = context::allocateMemory(workspace_size);
        workspace_ptr = workspace->data();
    }

    INFINICORE_CHECK_ERROR(infiniopDiagflat(
        desc,
        workspace_ptr,
        workspace_size,
        output->data(),
        input->data(),
        context::getStream()));
}

static bool registered = []() {
    Diagflat::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::diagflat_impl::infiniop


