#include "infinicore/ops/dsv4_sglang_hash_topk.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Dsv4SglangHashTopk);

Dsv4SglangHashTopk::Dsv4SglangHashTopk(const Tensor &router_logits, const Tensor &input_ids, const Tensor &tid2eid, Tensor topk_weights, Tensor topk_ids, float routed_scaling_factor) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(router_logits, input_ids, tid2eid, topk_weights, topk_ids);
    INFINICORE_GRAPH_OP_DISPATCH(router_logits->device().getType(), router_logits, input_ids, tid2eid, topk_weights, topk_ids, routed_scaling_factor);
}

void Dsv4SglangHashTopk::execute(const Tensor &router_logits, const Tensor &input_ids, const Tensor &tid2eid, Tensor topk_weights, Tensor topk_ids, float routed_scaling_factor) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Dsv4SglangHashTopk, router_logits, input_ids, tid2eid, topk_weights, topk_ids, routed_scaling_factor);
}

void dsv4_sglang_hash_topk_(const Tensor &router_logits, const Tensor &input_ids, const Tensor &tid2eid, Tensor topk_weights, Tensor topk_ids, float routed_scaling_factor) {
    Dsv4SglangHashTopk::execute(router_logits, input_ids, tid2eid, topk_weights, topk_ids, routed_scaling_factor);
}

} // namespace infinicore::op
