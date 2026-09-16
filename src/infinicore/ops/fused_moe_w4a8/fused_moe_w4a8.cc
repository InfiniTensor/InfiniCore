#include "infinicore/ops/fused_moe_w4a8.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(FusedMoeW4A8);

FusedMoeW4A8::FusedMoeW4A8(Tensor output,
                           const Tensor &input,
                           const Tensor &selected_experts,
                           const Tensor &routing_weights,
                           const Tensor &w13_packed,
                           const Tensor &w13_scale,
                           const Tensor &w2_packed,
                           const Tensor &w2_scale,
                           FusedMoeActivation activation,
                           bool weights_are_aiter_shuffled) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(
        output, input, selected_experts, routing_weights,
        w13_packed, w13_scale, w2_packed, w2_scale);
    INFINICORE_GRAPH_OP_DISPATCH(
        output->device().getType(), output, input, selected_experts,
        routing_weights, w13_packed, w13_scale, w2_packed, w2_scale,
        activation, weights_are_aiter_shuffled);
}

void FusedMoeW4A8::execute(Tensor output,
                           const Tensor &input,
                           const Tensor &selected_experts,
                           const Tensor &routing_weights,
                           const Tensor &w13_packed,
                           const Tensor &w13_scale,
                           const Tensor &w2_packed,
                           const Tensor &w2_scale,
                           FusedMoeActivation activation,
                           bool weights_are_aiter_shuffled) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(
        FusedMoeW4A8, output, input, selected_experts, routing_weights,
        w13_packed, w13_scale, w2_packed, w2_scale, activation,
        weights_are_aiter_shuffled);
}

Tensor fused_moe_w4a8(const Tensor &input,
                      const Tensor &selected_experts,
                      const Tensor &routing_weights,
                      const Tensor &w13_packed,
                      const Tensor &w13_scale,
                      const Tensor &w2_packed,
                      const Tensor &w2_scale,
                      FusedMoeActivation activation,
                      bool weights_are_aiter_shuffled) {
    auto output = Tensor::empty(input->shape(), input->dtype(), input->device());
    fused_moe_w4a8_(output, input, selected_experts, routing_weights,
                    w13_packed, w13_scale, w2_packed, w2_scale, activation,
                    weights_are_aiter_shuffled);
    return output;
}

void fused_moe_w4a8_(Tensor output,
                     const Tensor &input,
                     const Tensor &selected_experts,
                     const Tensor &routing_weights,
                     const Tensor &w13_packed,
                     const Tensor &w13_scale,
                     const Tensor &w2_packed,
                     const Tensor &w2_scale,
                     FusedMoeActivation activation,
                     bool weights_are_aiter_shuffled) {
    FusedMoeW4A8::execute(
        output, input, selected_experts, routing_weights,
        w13_packed, w13_scale, w2_packed, w2_scale, activation,
        weights_are_aiter_shuffled);
}

} // namespace infinicore::op
