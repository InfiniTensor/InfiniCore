#if defined(ENABLE_ATEN) && defined(ENABLE_HYGON_API)

#include "../../vendor_ops/vendor_ops_dispatch.hpp"

#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/adaptor/hygon_aiter_adaptor.hpp"

#include <optional>
#include <stdexcept>

namespace infinicore::op::grouped_topk_impl::hygon {

void run(Tensor topk_weights,
         Tensor topk_ids,
         const Tensor &scores,
         int64_t num_expert_group,
         int64_t topk_group,
         bool renormalize,
         const Tensor &bias,
         const std::string &scoring_func) {
    if (!adaptor::hygon_aiter::grouped_topk_available()) {
        throw std::runtime_error(
            "grouped_topk_vendor requires AITER grouped top-k support");
    }

    auto topk_weights_at = adaptor::to_aten_tensor(topk_weights);
    auto topk_ids_at = adaptor::to_aten_tensor(topk_ids);
    auto scores_at = adaptor::to_aten_tensor(scores);
    std::optional<at::Tensor> bias_at = std::nullopt;
    if (bias) {
        bias_at = adaptor::to_aten_tensor(bias);
    }
    adaptor::hygon_aiter::grouped_topk(
        topk_weights_at,
        topk_ids_at,
        scores_at,
        num_expert_group,
        topk_group,
        renormalize,
        bias_at,
        scoring_func == "softmax");
}

static bool registered = []() {
    vendor_ops::grouped_topk_dispatcher().registerDevice(Device::Type::HYGON, &run);
    return true;
}();

} // namespace infinicore::op::grouped_topk_impl::hygon

#endif
