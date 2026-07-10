#include "infinicore/ops/deepseek_v4_swa_decode.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(DeepseekV4SwaDecode);

DeepseekV4SwaDecode::DeepseekV4SwaDecode(Tensor y,
                                         const Tensor &q,
                                         const Tensor &k,
                                         const Tensor &attn_sink,
                                         float softmax_scale) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(y, q, k, attn_sink);
    INFINICORE_GRAPH_OP_DISPATCH(
        y->device().getType(), y, q, k, attn_sink, softmax_scale);
}

void DeepseekV4SwaDecode::execute(Tensor y,
                                  const Tensor &q,
                                  const Tensor &k,
                                  const Tensor &attn_sink,
                                  float softmax_scale) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(
        DeepseekV4SwaDecode, y, q, k, attn_sink, softmax_scale);
}

Tensor deepseek_v4_swa_decode(const Tensor &q,
                              const Tensor &k,
                              const Tensor &attn_sink,
                              float softmax_scale) {
    const auto &q_shape = q->shape();
    INFINICORE_ASSERT(q_shape.size() == 4);
    auto y = Tensor::empty(q_shape, q->dtype(), q->device());
    deepseek_v4_swa_decode_(y, q, k, attn_sink, softmax_scale);
    return y;
}

void deepseek_v4_swa_decode_(Tensor y,
                             const Tensor &q,
                             const Tensor &k,
                             const Tensor &attn_sink,
                             float softmax_scale) {
    DeepseekV4SwaDecode::execute(y, q, k, attn_sink, softmax_scale);
}

} // namespace infinicore::op
