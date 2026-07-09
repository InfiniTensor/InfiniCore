#include "infinicore/ops/dsv4_sglang_compress_c4_v2_prefill.hpp"
#include "../../utils.hpp"
namespace infinicore::op {
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Dsv4SglangCompressC4V2Prefill);
Dsv4SglangCompressC4V2Prefill::Dsv4SglangCompressC4V2Prefill(const Tensor &kv_buffer, const Tensor &kv_input, Tensor kv_output, const Tensor &ape, const Tensor &plan_c, const Tensor &plan_w) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(kv_buffer, kv_input, kv_output, ape, plan_c, plan_w);
    INFINICORE_GRAPH_OP_DISPATCH(kv_buffer->device().getType(), kv_buffer, kv_input, kv_output, ape, plan_c, plan_w);
}
void Dsv4SglangCompressC4V2Prefill::execute(const Tensor &kv_buffer, const Tensor &kv_input, Tensor kv_output, const Tensor &ape, const Tensor &plan_c, const Tensor &plan_w) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Dsv4SglangCompressC4V2Prefill, kv_buffer, kv_input, kv_output, ape, plan_c, plan_w);
}
void dsv4_sglang_compress_c4_v2_prefill_(const Tensor &kv_buffer, const Tensor &kv_input, Tensor kv_output, const Tensor &ape, const Tensor &plan_c, const Tensor &plan_w) {
    Dsv4SglangCompressC4V2Prefill::execute(kv_buffer, kv_input, kv_output, ape, plan_c, plan_w);
}
} // namespace infinicore::op
