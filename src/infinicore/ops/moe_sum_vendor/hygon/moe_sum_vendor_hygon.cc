#if defined(ENABLE_HYGON_API)
#include "../../infiniop_impl.hpp"
#include "../../vendor_ops/vendor_ops_dispatch.hpp"

#include "infinicore/context/context.hpp"
#include "infiniop/ops/moe_weighted_sum.h"

namespace infinicore::op::moe_sum_impl::hygon {

void run(Tensor output,
         const Tensor &input,
         std::optional<Tensor> topk_weights,
         std::optional<Tensor> extra_residual,
         double routed_scale,
         double residual_scale) {
    INFINICORE_CHECK_ERROR(infiniopMoeWeightedSum(
        context::getInfiniopHandle(output->device()),
        output->desc(),
        input->desc(),
        topk_weights ? (*topk_weights)->desc() : nullptr,
        extra_residual ? (*extra_residual)->desc() : nullptr,
        output->data(),
        input->data(),
        topk_weights ? (*topk_weights)->data() : nullptr,
        extra_residual ? (*extra_residual)->data() : nullptr,
        routed_scale,
        residual_scale,
        context::getStream()));
}

static bool registered = []() {
    vendor_ops::moe_sum_dispatcher().registerDevice(Device::Type::HYGON, &run);
    return true;
}();

} // namespace infinicore::op::moe_sum_impl::hygon
#endif
