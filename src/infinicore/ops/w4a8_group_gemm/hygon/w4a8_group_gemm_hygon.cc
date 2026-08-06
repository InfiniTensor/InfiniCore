#if defined(ENABLE_HYGON_API)
#include "../../infiniop_impl.hpp"
#include "../../vendor_ops/vendor_ops_dispatch.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/ops/w4a8_group_gemm.hpp"
#include "infiniop/ops/w4a8_group_gemm.h"

#include <stdexcept>

namespace infinicore::op::w4a8_group_gemm_impl::hygon {

void run(Tensor out,
         const Tensor &input,
         const Tensor &weight,
         const Tensor &input_scale,
         const Tensor &weight_scale,
         const Tensor &tokens_per_experts,
         std::optional<Tensor> sorted_token_ids,
         std::optional<Tensor> bias,
         bool trans_weight,
         bool) {
    if (tokens_per_experts->device() != out->device()) {
        throw std::runtime_error(
            "Hygon w4a8_group_gemm requires device-resident tokens_per_experts");
    }
    if (sorted_token_ids && (*sorted_token_ids)->device() != out->device()) {
        throw std::runtime_error(
            "Hygon w4a8_group_gemm requires device-resident sorted_token_ids");
    }
    infiniopW4A8GroupGemmDescriptor_t descriptor = nullptr;
    INFINICORE_CHECK_ERROR(infiniopCreateW4A8GroupGemmDescriptor(
        context::getInfiniopHandle(out->device()),
        &descriptor,
        out->desc(),
        input->desc(),
        weight->desc(),
        input_scale->desc(),
        weight_scale->desc(),
        tokens_per_experts->desc(),
        sorted_token_ids ? (*sorted_token_ids)->desc() : nullptr,
        bias ? (*bias)->desc() : nullptr,
        trans_weight));
    const auto status = infiniopW4A8GroupGemm(
        descriptor,
        out->data(),
        input->data(),
        weight->data(),
        input_scale->data(),
        weight_scale->data(),
        tokens_per_experts->data(),
        sorted_token_ids ? (*sorted_token_ids)->data() : nullptr,
        bias ? (*bias)->data() : nullptr,
        context::getStream());
    const auto destroy_status = infiniopDestroyW4A8GroupGemmDescriptor(descriptor);
    INFINICORE_CHECK_ERROR(status);
    INFINICORE_CHECK_ERROR(destroy_status);
}

static bool registered = []() {
    vendor_ops::w4a8_group_gemm_dispatcher().registerDevice(Device::Type::HYGON, &run);
    return true;
}();

} // namespace infinicore::op::w4a8_group_gemm_impl::hygon
#endif
