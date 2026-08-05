#if defined(ENABLE_ILUVATAR_VENDOR_OPS)
#include "../../vendor_ops/vendor_ops_dispatch.hpp"

#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/adaptor/iluvatar_vendor_adaptor.hpp"

#include <stdexcept>

namespace infinicore::op::moe_topk_impl::iluvatar {

void run_softmax(Tensor topk_weights,
                 Tensor topk_ids,
                 Tensor token_expert_indices,
                 const Tensor &gating_output,
                 bool renormalize,
                 const Tensor &correction_bias) {
    if (!adaptor::iluvatar_vendor::topk_softmax_available()) {
        throw std::runtime_error("moe_topk_softmax_vendor requires the Iluvatar vendor extension");
    }
    std::optional<at::Tensor> bias_at;
    if (correction_bias) {
        bias_at = adaptor::to_aten_tensor(correction_bias);
    }
    auto topk_weights_at = adaptor::to_aten_tensor(topk_weights);
    auto topk_ids_at = adaptor::to_aten_tensor(topk_ids);
    auto token_expert_indices_at = adaptor::to_aten_tensor(token_expert_indices);
    auto gating_output_at = adaptor::to_aten_tensor(gating_output);
    adaptor::iluvatar_vendor::topk_softmax(
        topk_weights_at, topk_ids_at,
        token_expert_indices_at,
        gating_output_at, renormalize, bias_at);
}

void run_sigmoid(Tensor topk_weights,
                 Tensor topk_ids,
                 Tensor token_expert_indices,
                 const Tensor &gating_output,
                 bool renormalize,
                 const Tensor &correction_bias) {
    if (!adaptor::iluvatar_vendor::topk_sigmoid_available()) {
        throw std::runtime_error("moe_topk_sigmoid_vendor requires the Iluvatar vendor extension");
    }
    std::optional<at::Tensor> bias_at;
    if (correction_bias) {
        bias_at = adaptor::to_aten_tensor(correction_bias);
    }
    auto topk_weights_at = adaptor::to_aten_tensor(topk_weights);
    auto topk_ids_at = adaptor::to_aten_tensor(topk_ids);
    auto token_expert_indices_at = adaptor::to_aten_tensor(token_expert_indices);
    auto gating_output_at = adaptor::to_aten_tensor(gating_output);
    adaptor::iluvatar_vendor::topk_sigmoid(
        topk_weights_at, topk_ids_at,
        token_expert_indices_at,
        gating_output_at, renormalize, bias_at);
}

static bool registered = []() {
    vendor_ops::moe_topk_softmax_dispatcher().registerDevice(Device::Type::ILUVATAR, &run_softmax);
    vendor_ops::moe_topk_sigmoid_dispatcher().registerDevice(Device::Type::ILUVATAR, &run_sigmoid);
    return true;
}();

} // namespace infinicore::op::moe_topk_impl::iluvatar
#endif
