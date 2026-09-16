#include "infinicore/ops/linear_w4a8.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(LinearW4A8);

LinearW4A8::LinearW4A8(Tensor output,
                       const Tensor &input,
                       const Tensor &packed_weight,
                       const Tensor &weight_scale,
                       std::optional<Tensor> bias,
                       float alpha) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(
        output, input, packed_weight, weight_scale);
    if (bias.has_value()) {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, bias.value());
    }
    INFINICORE_GRAPH_OP_DISPATCH(
        output->device().getType(), output, input, packed_weight,
        weight_scale, bias, alpha);
}

void LinearW4A8::execute(Tensor output,
                         const Tensor &input,
                         const Tensor &packed_weight,
                         const Tensor &weight_scale,
                         std::optional<Tensor> bias,
                         float alpha) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(
        LinearW4A8, output, input, packed_weight, weight_scale, bias, alpha);
}

Tensor linear_w4a8(const Tensor &input,
                   const Tensor &packed_weight,
                   const Tensor &weight_scale,
                   std::optional<Tensor> bias,
                   float alpha) {
    INFINICORE_ASSERT(input->ndim() >= 2);
    INFINICORE_ASSERT(packed_weight->ndim() == 2);
    auto output_shape = input->shape();
    output_shape.back() = packed_weight->size(0);
    auto output = Tensor::empty(output_shape, input->dtype(), input->device());
    linear_w4a8_(output, input, packed_weight, weight_scale, bias, alpha);
    return output;
}

void linear_w4a8_(Tensor output,
                  const Tensor &input,
                  const Tensor &packed_weight,
                  const Tensor &weight_scale,
                  std::optional<Tensor> bias,
                  float alpha) {
    LinearW4A8::execute(
        output, input, packed_weight, weight_scale, bias, alpha);
}

} // namespace infinicore::op
