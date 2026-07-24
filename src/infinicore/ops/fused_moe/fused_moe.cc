#include "infinicore/ops/fused_moe.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(FusedMoe);

FusedMoe::FusedMoe(Tensor out,
                   const Tensor &input,
                   const Tensor &token_selected_experts,
                   const Tensor &token_final_scales,
                   const Tensor &w1,
                   const Tensor &w2,
                   std::optional<Tensor> b1,
                   std::optional<Tensor> b2,
                   FusedMoeActivation activation) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, input, token_selected_experts, token_final_scales, w1, w2);
    if (b1.has_value()) {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, b1.value());
    }
    if (b2.has_value()) {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, b2.value());
    }
    INFINICORE_GRAPH_OP_DISPATCH(out->device().getType(), out, input, token_selected_experts,
                                 token_final_scales, w1, w2, b1, b2, activation);
}

void FusedMoe::execute(Tensor out,
                       const Tensor &input,
                       const Tensor &token_selected_experts,
                       const Tensor &token_final_scales,
                       const Tensor &w1,
                       const Tensor &w2,
                       std::optional<Tensor> b1,
                       std::optional<Tensor> b2,
                       FusedMoeActivation activation) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(FusedMoe, out, input, token_selected_experts,
                                      token_final_scales, w1, w2, b1, b2, activation);
}

Tensor fused_moe(const Tensor &input,
                 const Tensor &token_selected_experts,
                 const Tensor &token_final_scales,
                 const Tensor &w1,
                 const Tensor &w2,
                 std::optional<Tensor> b1,
                 std::optional<Tensor> b2,
                 FusedMoeActivation activation) {
    auto out = Tensor::empty(input->shape(), input->dtype(), input->device());
    fused_moe_(out, input, token_selected_experts, token_final_scales, w1, w2, b1, b2, activation);
    return out;
}

void fused_moe_(Tensor out,
                const Tensor &input,
                const Tensor &token_selected_experts,
                const Tensor &token_final_scales,
                const Tensor &w1,
                const Tensor &w2,
                std::optional<Tensor> b1,
                std::optional<Tensor> b2,
                FusedMoeActivation activation) {
    FusedMoe::execute(out, input, token_selected_experts, token_final_scales, w1, w2, b1, b2, activation);
}

} // namespace infinicore::op
