#include "infinicore/ops/kimi_attention_residual.hpp"

#include "../../utils.hpp"

#include <stdexcept>

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(KimiAttentionResidual);

KimiAttentionResidual::KimiAttentionResidual(
    Tensor output,
    const Tensor &values,
    const Tensor &norm_weight,
    const Tensor &projection_weight,
    float epsilon) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(
        output, values, norm_weight, projection_weight);
    INFINICORE_GRAPH_OP_DISPATCH(output->device().getType(),
                                 output,
                                 values,
                                 norm_weight,
                                 projection_weight,
                                 epsilon);
}

void KimiAttentionResidual::execute(
    Tensor output,
    const Tensor &values,
    const Tensor &norm_weight,
    const Tensor &projection_weight,
    float epsilon) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(KimiAttentionResidual,
                                      output,
                                      values,
                                      norm_weight,
                                      projection_weight,
                                      epsilon);
}

namespace {

void check_inputs(const Tensor &values,
                  const Tensor &norm_weight,
                  const Tensor &projection_weight) {
    if (values->shape().size() != 3) {
        throw std::runtime_error(
            "kimi_attention_residual expects values [tokens, sources, hidden]");
    }
    const size_t hidden_size = values->shape()[2];
    if (norm_weight->shape() != Shape{hidden_size}) {
        throw std::runtime_error(
            "kimi_attention_residual expects norm_weight [hidden]");
    }
    const auto &projection_shape = projection_weight->shape();
    if (projection_shape != Shape{hidden_size}
        && projection_shape != Shape{1, hidden_size}) {
        throw std::runtime_error(
            "kimi_attention_residual expects projection_weight [hidden] or [1, hidden]");
    }
}

} // namespace

Tensor kimi_attention_residual(const Tensor &values,
                               const Tensor &norm_weight,
                               const Tensor &projection_weight,
                               float epsilon) {
    check_inputs(values, norm_weight, projection_weight);
    auto output = Tensor::empty(
        {values->shape()[0], values->shape()[2]}, values->dtype(), values->device());
    kimi_attention_residual_(output, values, norm_weight, projection_weight, epsilon);
    return output;
}

void kimi_attention_residual_(Tensor output,
                              const Tensor &values,
                              const Tensor &norm_weight,
                              const Tensor &projection_weight,
                              float epsilon) {
    check_inputs(values, norm_weight, projection_weight);
    if (output->shape() != Shape{values->shape()[0], values->shape()[2]}) {
        throw std::runtime_error(
            "kimi_attention_residual_ expects output [tokens, hidden]");
    }
    KimiAttentionResidual::execute(
        output, values, norm_weight, projection_weight, epsilon);
}

} // namespace infinicore::op
