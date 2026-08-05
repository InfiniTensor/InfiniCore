#if defined(ENABLE_ILUVATAR_VENDOR_OPS)
#include "../../vendor_ops/vendor_ops_dispatch.hpp"

#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/adaptor/iluvatar_vendor_adaptor.hpp"

#include <stdexcept>

namespace infinicore::op::moe_sum_impl::iluvatar {

void run(Tensor output,
         const Tensor &input,
         std::optional<Tensor> topk_weights,
         std::optional<Tensor> extra_residual,
         double routed_scale,
         double residual_scale) {
    if (!adaptor::iluvatar_vendor::moe_sum_vendor_available()) {
        throw std::runtime_error("moe_sum_vendor requires the Iluvatar vendor extension");
    }
    std::optional<at::Tensor> weights_at;
    if (topk_weights) {
        weights_at = adaptor::to_aten_tensor(*topk_weights);
    }
    std::optional<at::Tensor> residual_at;
    if (extra_residual) {
        residual_at = adaptor::to_aten_tensor(*extra_residual);
    }
    auto output_at = adaptor::to_aten_tensor(output);
    auto input_at = adaptor::to_aten_tensor(input);
    adaptor::iluvatar_vendor::moe_sum_vendor(
        output_at, input_at,
        weights_at, residual_at, routed_scale, residual_scale);
}

static bool registered = []() {
    vendor_ops::moe_sum_dispatcher().registerDevice(Device::Type::ILUVATAR, &run);
    return true;
}();

} // namespace infinicore::op::moe_sum_impl::iluvatar
#endif
