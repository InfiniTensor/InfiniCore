#include "infinicore/ops/simple_gla_attention.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(SimpleGlaAttention);

SimpleGlaAttention::SimpleGlaAttention(Tensor out,
                                       const Tensor &q,
                                       const Tensor &k,
                                       const Tensor &v,
                                       const Tensor &g_gamma,
                                       float scale) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, q, k, v, g_gamma);
    INFINICORE_GRAPH_OP_DISPATCH(out->device().getType(), out, q, k, v, g_gamma, scale);
}

void SimpleGlaAttention::execute(Tensor out,
                                 const Tensor &q,
                                 const Tensor &k,
                                 const Tensor &v,
                                 const Tensor &g_gamma,
                                 float scale) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(SimpleGlaAttention, out, q, k, v, g_gamma, scale);
}

Tensor simple_gla_attention(const Tensor &q,
                            const Tensor &k,
                            const Tensor &v,
                            const Tensor &g_gamma,
                            float scale) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(q, k, v, g_gamma);

    const auto &q_shape = q->shape();
    INFINICORE_ASSERT(q_shape.size() == 4);
    INFINICORE_ASSERT(k->shape() == q_shape && v->shape() == q_shape);
    INFINICORE_ASSERT(g_gamma->shape().size() == 1 && g_gamma->shape()[0] == q_shape[2]);

    auto q_cont = q->contiguous();
    auto k_cont = k->contiguous();
    auto v_cont = v->contiguous();
    auto g_cont = g_gamma->contiguous();

    auto out = Tensor::empty(q_shape, q->dtype(), q->device());
    SimpleGlaAttention::execute(out, q_cont, k_cont, v_cont, g_cont, scale);
    return out;
}

} // namespace infinicore::op
