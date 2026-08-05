#if defined(ENABLE_ILUVATAR_VENDOR_OPS)
#include "../../vendor_ops/vendor_ops_dispatch.hpp"

#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/adaptor/iluvatar_vendor_adaptor.hpp"

#include <stdexcept>

namespace infinicore::op::moe_argsort_impl::iluvatar {

void run(Tensor tokens_per_experts,
         Tensor sorted_indices,
         Tensor inv_pos,
         const Tensor &topk_ids,
         int64_t num_experts) {
    if (!adaptor::iluvatar_vendor::argsort_bincount_with_inv_pos_available()) {
        throw std::runtime_error("moe_argsort_bincount_with_inv_pos requires the Iluvatar vendor extension");
    }
    auto topk_ids_at = adaptor::to_aten_tensor(topk_ids);
    auto tokens_per_experts_at = adaptor::to_aten_tensor(tokens_per_experts);
    auto sorted_indices_at = adaptor::to_aten_tensor(sorted_indices);
    auto inv_pos_at = adaptor::to_aten_tensor(inv_pos);
    adaptor::iluvatar_vendor::argsort_bincount_with_inv_pos(
        topk_ids_at, tokens_per_experts_at,
        sorted_indices_at, inv_pos_at,
        num_experts);
}

static bool registered = []() {
    vendor_ops::moe_argsort_dispatcher().registerDevice(Device::Type::ILUVATAR, &run);
    return true;
}();

} // namespace infinicore::op::moe_argsort_impl::iluvatar
#endif
