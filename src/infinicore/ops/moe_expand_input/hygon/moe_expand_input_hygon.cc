#if defined(ENABLE_HYGON_API)
#include "../../vendor_ops/vendor_ops_dispatch.hpp"

#include "infinicore/ops/broadcast_to.hpp"
#include "infinicore/ops/dynamic_scaled_int8_quant.hpp"
#include "infinicore/ops/index_copy.hpp"

#include <stdexcept>

namespace infinicore::op::moe_expand_input_impl::hygon {

void run(Tensor expand_states,
         std::optional<Tensor> expand_scales,
         const Tensor &hidden_states,
         const Tensor &inv_pos,
         int64_t top_k,
         int64_t,
         int64_t format) {
    const size_t tokens = hidden_states->size(0);
    const size_t hidden = hidden_states->size(1);
    const size_t total = tokens * static_cast<size_t>(top_k);
    if (expand_states->size(1) != hidden) {
        throw std::runtime_error(
            "Hygon moe_expand_input currently requires an unpadded hidden size");
    }
    if (format != 0 && format != 1) {
        throw std::runtime_error(
            "Hygon moe_expand_input supports normal or conventional INT8 format");
    }

    auto repeated = broadcast_to(
                        hidden_states->view({tokens, 1, hidden}),
                        {static_cast<int64_t>(tokens), top_k, static_cast<int64_t>(hidden)})
                        ->view({total, hidden});
    if (format == 0) {
        index_copy_(expand_states, expand_states, 0, inv_pos, repeated);
        return;
    }
    if (!expand_scales) {
        throw std::runtime_error(
            "Hygon quantized moe_expand_input requires scales");
    }
    auto grouped = Tensor::empty(
        {total, hidden}, hidden_states->dtype(), hidden_states->device());
    index_copy_(grouped, grouped, 0, inv_pos, repeated);
    dynamic_scaled_int8_quant_(expand_states, grouped, *expand_scales);
}

static bool registered = []() {
    vendor_ops::moe_expand_input_dispatcher().registerDevice(
        Device::Type::HYGON, &run);
    return true;
}();

} // namespace infinicore::op::moe_expand_input_impl::hygon
#endif
