#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/where.hpp"
#include <infiniop.h>

namespace infinicore::op::where_impl::infiniop {

thread_local common::OpCache<size_t, infiniopWhereDescriptor_t> caches(
    100,
    [](infiniopWhereDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyWhereDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor out, Tensor cond, Tensor x, Tensor y) {
    size_t seed = hash_combine(out, cond, x, y);

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();

    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopWhereDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateWhereDescriptor(
            context::getInfiniopHandle(out->device()), &desc,
            out->desc(), cond->desc(), x->desc(), y->desc()));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetWhereWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopWhere(
        desc, workspace->data(), workspace_size,
        out->data(), cond->data(), x->data(), y->data(), context::getStream()));
}

static bool registered = []() {
    Where::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::where_impl::infiniop
