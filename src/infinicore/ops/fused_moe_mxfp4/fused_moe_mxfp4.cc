#include "infinicore/ops/fused_moe_mxfp4.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(FusedMoeMxfp4);

FusedMoeMxfp4::FusedMoeMxfp4(Tensor output,
                             const Tensor &input,
                             const Tensor &selected_experts,
                             const Tensor &routing_weights,
                             const Tensor &w13_packed,
                             const Tensor &w13_scale,
                             const Tensor &w2_packed,
                             const Tensor &w2_scale,
                             FusedMoeActivation activation) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(
        output, input, selected_experts, routing_weights,
        w13_packed, w13_scale, w2_packed, w2_scale);
    INFINICORE_GRAPH_OP_DISPATCH(
        output->device().getType(), output, input, selected_experts, routing_weights,
        w13_packed, w13_scale, w2_packed, w2_scale, activation);
}

void FusedMoeMxfp4::execute(Tensor output,
                            const Tensor &input,
                            const Tensor &selected_experts,
                            const Tensor &routing_weights,
                            const Tensor &w13_packed,
                            const Tensor &w13_scale,
                            const Tensor &w2_packed,
                            const Tensor &w2_scale,
                            FusedMoeActivation activation) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(
        FusedMoeMxfp4, output, input, selected_experts, routing_weights,
        w13_packed, w13_scale, w2_packed, w2_scale, activation);
}

Tensor fused_moe_mxfp4(const Tensor &input,
                       const Tensor &selected_experts,
                       const Tensor &routing_weights,
                       const Tensor &w13_packed,
                       const Tensor &w13_scale,
                       const Tensor &w2_packed,
                       const Tensor &w2_scale,
                       FusedMoeActivation activation) {
    auto output = Tensor::empty(input->shape(), input->dtype(), input->device());
    fused_moe_mxfp4_(output, input, selected_experts, routing_weights,
                     w13_packed, w13_scale, w2_packed, w2_scale, activation);
    return output;
}

void fused_moe_mxfp4_(Tensor output,
                      const Tensor &input,
                      const Tensor &selected_experts,
                      const Tensor &routing_weights,
                      const Tensor &w13_packed,
                      const Tensor &w13_scale,
                      const Tensor &w2_packed,
                      const Tensor &w2_scale,
                      FusedMoeActivation activation) {
    FusedMoeMxfp4::execute(output, input, selected_experts, routing_weights,
                           w13_packed, w13_scale, w2_packed, w2_scale, activation);
}

} // namespace infinicore::op
