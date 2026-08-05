#if defined(ENABLE_ILUVATAR_VENDOR_OPS)
#include "../../vendor_ops/vendor_ops_dispatch.hpp"

#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/adaptor/iluvatar_vendor_adaptor.hpp"

#include <stdexcept>

namespace infinicore::op::grouped_topk_impl::iluvatar {

void run(Tensor topk_weights,
         Tensor topk_ids,
         const Tensor &scores,
         int64_t num_expert_group,
         int64_t topk_group,
         bool renormalize,
         const Tensor &bias,
         const std::string &scoring_func) {
    const auto experts = scores->size(1);
    if (num_expert_group != 1 && num_expert_group != 8) {
        throw std::runtime_error("grouped_topk_vendor Iluvatar implementation supports num_expert_group 1 or 8");
    }
    if (!(experts == 64 || experts == 128 || experts == 160 || experts == 192
          || experts == 256 || experts == 384)) {
        throw std::runtime_error("grouped_topk_vendor Iluvatar implementation supports expert counts 64/128/160/192/256/384");
    }
    if (scores->dtype() != DataType::F16 && scores->dtype() != DataType::BF16) {
        throw std::runtime_error("grouped_topk_vendor Iluvatar implementation supports fp16/bfloat16 scores");
    }
    if (!bias) {
        throw std::runtime_error("grouped_topk_vendor Iluvatar implementation requires correction bias");
    }
    if (!adaptor::iluvatar_vendor::grouped_topk_available()) {
        throw std::runtime_error("grouped_topk_vendor requires the Iluvatar vendor extension");
    }
    std::optional<at::Tensor> bias_at = adaptor::to_aten_tensor(bias);
    auto topk_weights_at = adaptor::to_aten_tensor(topk_weights);
    auto topk_ids_at = adaptor::to_aten_tensor(topk_ids);
    auto scores_at = adaptor::to_aten_tensor(scores);
    adaptor::iluvatar_vendor::grouped_topk(
        topk_weights_at, topk_ids_at,
        scores_at, bias_at, num_expert_group, topk_group,
        scoring_func, renormalize);
}

static bool registered = []() {
    vendor_ops::grouped_topk_dispatcher().registerDevice(Device::Type::ILUVATAR, &run);
    return true;
}();

} // namespace infinicore::op::grouped_topk_impl::iluvatar
#endif
