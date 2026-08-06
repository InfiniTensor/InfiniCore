#if defined(ENABLE_HYGON_API)
#include "../../vendor_ops/vendor_ops_dispatch.hpp"

#include "infinicore/ops/moe_topk_sigmoid.hpp"

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
    if (num_expert_group != 1 || topk_group != 1) {
        throw std::runtime_error(
            "Hygon grouped_topk currently supports a single expert group");
    }
    if (scoring_func != "sigmoid") {
        throw std::runtime_error(
            "Hygon grouped_topk currently supports sigmoid scoring");
    }
    if (!bias || bias->dtype() != DataType::F32) {
        throw std::runtime_error(
            "Hygon grouped_topk requires float32 correction bias");
    }
    moe_topk_sigmoid_(
        topk_weights, topk_ids, scores, bias, renormalize);
}

static bool registered = []() {
    vendor_ops::grouped_topk_dispatcher().registerDevice(Device::Type::HYGON, &run);
    return true;
}();

} // namespace infinicore::op::grouped_topk_impl::hygon
#endif
