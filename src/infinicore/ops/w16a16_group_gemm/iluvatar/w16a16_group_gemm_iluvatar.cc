#if defined(ENABLE_ILUVATAR_VENDOR_OPS)
#include "../../vendor_ops/vendor_ops_dispatch.hpp"

#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/adaptor/iluvatar_vendor_adaptor.hpp"

#include <stdexcept>

namespace infinicore::op::w16a16_group_gemm_impl::iluvatar {

void run(Tensor out,
         const Tensor &input,
         const Tensor &weight,
         const Tensor &tokens_per_experts,
         std::optional<Tensor> sorted_token_ids,
         std::optional<Tensor> bias,
         bool trans_weight,
         bool is_decode) {
    if (!adaptor::iluvatar_vendor::w16a16_group_gemm_available()) {
        throw std::runtime_error("w16a16_group_gemm requires the Iluvatar vendor extension");
    }
    std::optional<at::Tensor> sorted_at;
    if (sorted_token_ids) {
        sorted_at = adaptor::to_aten_tensor(*sorted_token_ids);
    }
    std::optional<at::Tensor> bias_at;
    if (bias) {
        bias_at = adaptor::to_aten_tensor(*bias);
    }
    auto out_at = adaptor::to_aten_tensor(out);
    auto input_at = adaptor::to_aten_tensor(input);
    auto weight_at = adaptor::to_aten_tensor(weight);
    auto tokens_per_experts_at = adaptor::to_aten_tensor(tokens_per_experts);
    adaptor::iluvatar_vendor::w16a16_group_gemm(
        out_at, input_at,
        weight_at, tokens_per_experts_at,
        sorted_at, bias_at, trans_weight, is_decode);
}

static bool registered = []() {
    vendor_ops::w16a16_group_gemm_dispatcher().registerDevice(Device::Type::ILUVATAR, &run);
    return true;
}();

} // namespace infinicore::op::w16a16_group_gemm_impl::iluvatar
#endif
