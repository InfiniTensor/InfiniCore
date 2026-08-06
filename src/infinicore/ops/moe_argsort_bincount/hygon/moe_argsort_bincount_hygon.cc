#if defined(ENABLE_HYGON_API)
#include "../../infiniop_impl.hpp"
#include "../../vendor_ops/vendor_ops_dispatch.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/ops/moe_argsort_bincount.hpp"
#include "infiniop/ops/moe_argsort_bincount.h"

namespace infinicore::op::moe_argsort_impl::hygon {

void run(Tensor tokens_per_experts,
         Tensor sorted_indices,
         Tensor inv_pos,
         const Tensor &topk_ids,
         int64_t num_experts) {
    infiniopMoeArgsortBincountDescriptor_t descriptor = nullptr;
    INFINICORE_CHECK_ERROR(infiniopCreateMoeArgsortBincountDescriptor(
        context::getInfiniopHandle(topk_ids->device()),
        &descriptor,
        tokens_per_experts->desc(),
        sorted_indices->desc(),
        inv_pos->desc(),
        topk_ids->desc(),
        static_cast<size_t>(num_experts)));
    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetMoeArgsortBincountWorkspaceSize(
        descriptor, &workspace_size));
    auto workspace = context::allocateMemory(workspace_size);
    const auto status = infiniopMoeArgsortBincount(
        descriptor,
        workspace->data(),
        workspace_size,
        tokens_per_experts->data(),
        sorted_indices->data(),
        inv_pos->data(),
        topk_ids->data(),
        context::getStream());
    const auto destroy_status = infiniopDestroyMoeArgsortBincountDescriptor(descriptor);
    INFINICORE_CHECK_ERROR(status);
    INFINICORE_CHECK_ERROR(destroy_status);
}

static bool registered = []() {
    vendor_ops::moe_argsort_dispatcher().registerDevice(
        Device::Type::HYGON, &run);
    return true;
}();

} // namespace infinicore::op::moe_argsort_impl::hygon
#endif
