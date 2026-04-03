#include "infinicore/ops/simple_gla_prefill.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(SimpleGLAPrefill);

SimpleGLAPrefill::SimpleGLAPrefill(Tensor out,
                                   const Tensor &q,
                                   const Tensor &k,
                                   const Tensor &v,
                                   const Tensor &g_gamma,
                                   float scale) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, q, k, v, g_gamma);
    INFINICORE_GRAPH_OP_DISPATCH(out->device().getType(), out, q, k, v, g_gamma, scale);
}

void SimpleGLAPrefill::execute(Tensor out,
                               const Tensor &q,
                               const Tensor &k,
                               const Tensor &v,
                               const Tensor &g_gamma,
                               float scale) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(SimpleGLAPrefill, out, q, k, v, g_gamma, scale);
}

Tensor simple_gla_prefill(const Tensor &q,
                          const Tensor &k,
                          const Tensor &v,
                          const Tensor &g_gamma,
                          float scale) {
    auto out = Tensor::empty(q->shape(), q->dtype(), q->device());
    SimpleGLAPrefill::execute(out, q, k, v, g_gamma, scale);
    return out;
}

} // namespace infinicore::op

