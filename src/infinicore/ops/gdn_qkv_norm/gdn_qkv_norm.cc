#include "infinicore/ops/gdn_qkv_norm.hpp"
#include "../../utils.hpp"
namespace infinicore::op {
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(GDNQKVNorm);
GDNQKVNorm::GDNQKVNorm(Tensor q_out, Tensor k_out, Tensor v_out, const Tensor &q, const Tensor &k, const Tensor &v) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(q_out, k_out, v_out, q, k, v);
    INFINICORE_GRAPH_OP_DISPATCH(q->device().getType(), q_out, k_out, v_out, q, k, v);
}
void GDNQKVNorm::execute(Tensor q_out, Tensor k_out, Tensor v_out, const Tensor &q, const Tensor &k, const Tensor &v) { INFINICORE_GRAPH_OP_RECORD_OR_RUN(GDNQKVNorm, q_out, k_out, v_out, q, k, v); }
void gdn_qkv_norm_(Tensor q_out, Tensor k_out, Tensor v_out, const Tensor &q, const Tensor &k, const Tensor &v) { GDNQKVNorm::execute(q_out, k_out, v_out, q, k, v); }
} // namespace infinicore::op
