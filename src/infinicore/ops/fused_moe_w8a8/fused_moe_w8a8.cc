#include "infinicore/ops/fused_moe_w8a8.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(FusedMoeW8A8);

FusedMoeW8A8::FusedMoeW8A8(Tensor output,
                           const Tensor &input,
                           const Tensor &selected_experts,
                           const Tensor &routing_weights,
                           const Tensor &w1,
                           const Tensor &w1_scale,
                           const Tensor &w2,
                           const Tensor &w2_scale) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(
        output, input, selected_experts, routing_weights,
        w1, w1_scale, w2, w2_scale);
    INFINICORE_GRAPH_OP_DISPATCH(
        output->device().getType(), output, input, selected_experts,
        routing_weights, w1, w1_scale, w2, w2_scale);
}

void FusedMoeW8A8::execute(Tensor output,
                           const Tensor &input,
                           const Tensor &selected_experts,
                           const Tensor &routing_weights,
                           const Tensor &w1,
                           const Tensor &w1_scale,
                           const Tensor &w2,
                           const Tensor &w2_scale) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(
        FusedMoeW8A8, output, input, selected_experts, routing_weights,
        w1, w1_scale, w2, w2_scale);
}

Tensor fused_moe_w8a8(const Tensor &input,
                      const Tensor &selected_experts,
                      const Tensor &routing_weights,
                      const Tensor &w1,
                      const Tensor &w1_scale,
                      const Tensor &w2,
                      const Tensor &w2_scale) {
    auto output = Tensor::empty(input->shape(), input->dtype(), input->device());
    fused_moe_w8a8_(output, input, selected_experts, routing_weights,
                    w1, w1_scale, w2, w2_scale);
    return output;
}

void fused_moe_w8a8_(Tensor output,
                     const Tensor &input,
                     const Tensor &selected_experts,
                     const Tensor &routing_weights,
                     const Tensor &w1,
                     const Tensor &w1_scale,
                     const Tensor &w2,
                     const Tensor &w2_scale) {
    FusedMoeW8A8::execute(output, input, selected_experts, routing_weights,
                          w1, w1_scale, w2, w2_scale);
}

} // namespace infinicore::op
