#include "infinicore/ops/linear_mxfp4.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(LinearMxfp4);

LinearMxfp4::LinearMxfp4(Tensor output,
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
        output->device().getType(), output, input, packed_weight, weight_scale, bias, alpha);
}

void LinearMxfp4::execute(Tensor output,
                          const Tensor &input,
                          const Tensor &packed_weight,
                          const Tensor &weight_scale,
                          std::optional<Tensor> bias,
                          float alpha) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(
        LinearMxfp4, output, input, packed_weight, weight_scale, bias, alpha);
}

Tensor linear_mxfp4(const Tensor &input,
                    const Tensor &packed_weight,
                    const Tensor &weight_scale,
                    std::optional<Tensor> bias,
                    float alpha) {
    INFINICORE_ASSERT(input->ndim() >= 2);
    INFINICORE_ASSERT(packed_weight->ndim() == 2);
    auto output_shape = input->shape();
    output_shape.back() = packed_weight->size(0);
    auto output = Tensor::empty(output_shape, input->dtype(), input->device());
    linear_mxfp4_(output, input, packed_weight, weight_scale, bias, alpha);
    return output;
}

void linear_mxfp4_(Tensor output,
                   const Tensor &input,
                   const Tensor &packed_weight,
                   const Tensor &weight_scale,
                   std::optional<Tensor> bias,
                   float alpha) {
    LinearMxfp4::execute(output, input, packed_weight, weight_scale, bias, alpha);
}

} // namespace infinicore::op
