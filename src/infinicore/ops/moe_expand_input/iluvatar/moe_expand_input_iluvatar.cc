#if defined(ENABLE_ILUVATAR_VENDOR_OPS)
#include "../../vendor_ops/vendor_ops_dispatch.hpp"

#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/adaptor/iluvatar_vendor_adaptor.hpp"

#include <stdexcept>

namespace infinicore::op::moe_expand_input_impl::iluvatar {

void run(Tensor expand_states,
         std::optional<Tensor> expand_scales,
         const Tensor &hidden_states,
         const Tensor &inv_pos,
         int64_t top_k,
         int64_t group_size,
         int64_t format) {
    if (!adaptor::iluvatar_vendor::expand_moe_input_with_inv_pos_available()) {
        throw std::runtime_error("moe_expand_input_with_inv_pos requires the Iluvatar vendor extension");
    }
    std::optional<at::Tensor> scales_at;
    if (expand_scales) {
        scales_at = adaptor::to_aten_tensor(*expand_scales);
    }
    auto expand_states_at = adaptor::to_aten_tensor(expand_states);
    auto hidden_states_at = adaptor::to_aten_tensor(hidden_states);
    auto inv_pos_at = adaptor::to_aten_tensor(inv_pos);
    adaptor::iluvatar_vendor::expand_moe_input_with_inv_pos(
        expand_states_at, scales_at,
        hidden_states_at, inv_pos_at,
        top_k, group_size, format);
}

static bool registered = []() {
    vendor_ops::moe_expand_input_dispatcher().registerDevice(Device::Type::ILUVATAR, &run);
    return true;
}();

} // namespace infinicore::op::moe_expand_input_impl::iluvatar
#endif
