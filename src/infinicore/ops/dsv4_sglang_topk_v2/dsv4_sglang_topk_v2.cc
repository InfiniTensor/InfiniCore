#include "infinicore/ops/dsv4_sglang_topk_v2.hpp"

#include "../../utils.hpp"
namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Dsv4SglangTopkV2);

Dsv4SglangTopkV2::Dsv4SglangTopkV2(const Tensor &scores, const Tensor &seq_lens, const Tensor &page_table, Tensor page_indices, Tensor transform_workspace, Tensor metadata, int64_t page_size) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(scores, seq_lens, page_table, page_indices, transform_workspace, metadata);
    INFINICORE_GRAPH_OP_DISPATCH(scores->device().getType(), scores, seq_lens, page_table, page_indices, transform_workspace, metadata, page_size);
}

void Dsv4SglangTopkV2::execute(const Tensor &scores, const Tensor &seq_lens, const Tensor &page_table, Tensor page_indices, Tensor transform_workspace, Tensor metadata, int64_t page_size) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Dsv4SglangTopkV2, scores, seq_lens, page_table, page_indices, transform_workspace, metadata, page_size);
}

void dsv4_sglang_topk_v2_(const Tensor &scores, const Tensor &seq_lens, const Tensor &page_table, Tensor page_indices, Tensor transform_workspace, Tensor metadata, int64_t page_size) {
    Dsv4SglangTopkV2::execute(scores, seq_lens, page_table, page_indices, transform_workspace, metadata, page_size);
}

} // namespace infinicore::op
