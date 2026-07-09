#include "infinicore/ops/dsv4_sglang_compress_c128_online_v2_prefill.hpp"
#include "../../utils.hpp"
namespace infinicore::op {
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Dsv4SglangCompressC128OnlineV2Prefill);
Dsv4SglangCompressC128OnlineV2Prefill::Dsv4SglangCompressC128OnlineV2Prefill(const Tensor &kv_score_buffer, const Tensor &kv_score_input, Tensor kv_output, const Tensor &ape, const Tensor &plan_c, const Tensor &plan_w) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(kv_score_buffer, kv_score_input, kv_output, ape, plan_c, plan_w);
    INFINICORE_GRAPH_OP_DISPATCH(kv_score_buffer->device().getType(), kv_score_buffer, kv_score_input, kv_output, ape, plan_c, plan_w);
}
void Dsv4SglangCompressC128OnlineV2Prefill::execute(const Tensor &kv_score_buffer, const Tensor &kv_score_input, Tensor kv_output, const Tensor &ape, const Tensor &plan_c, const Tensor &plan_w) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Dsv4SglangCompressC128OnlineV2Prefill, kv_score_buffer, kv_score_input, kv_output, ape, plan_c, plan_w);
}
void dsv4_sglang_compress_c128_online_v2_prefill_(const Tensor &kv_score_buffer, const Tensor &kv_score_input, Tensor kv_output, const Tensor &ape, const Tensor &plan_c, const Tensor &plan_w) {
    Dsv4SglangCompressC128OnlineV2Prefill::execute(kv_score_buffer, kv_score_input, kv_output, ape, plan_c, plan_w);
}
} // namespace infinicore::op
