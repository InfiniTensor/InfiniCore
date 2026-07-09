#include "infinicore/ops/dsv4_sglang_topk_transform.hpp"

#include "../../utils.hpp"
namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Dsv4SglangTopkTransform);

Dsv4SglangTopkTransform::Dsv4SglangTopkTransform(const Tensor &scores, const Tensor &seq_lens, const Tensor &page_table, Tensor page_indices, Tensor raw_indices, int64_t page_size) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(scores, seq_lens, page_table, page_indices, raw_indices);
    INFINICORE_GRAPH_OP_DISPATCH(scores->device().getType(), scores, seq_lens, page_table, page_indices, raw_indices, page_size);
}

void Dsv4SglangTopkTransform::execute(const Tensor &scores, const Tensor &seq_lens, const Tensor &page_table, Tensor page_indices, Tensor raw_indices, int64_t page_size) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Dsv4SglangTopkTransform, scores, seq_lens, page_table, page_indices, raw_indices, page_size);
}

void dsv4_sglang_topk_transform_(const Tensor &scores, const Tensor &seq_lens, const Tensor &page_table, Tensor page_indices, Tensor raw_indices, int64_t page_size) {
    Dsv4SglangTopkTransform::execute(scores, seq_lens, page_table, page_indices, raw_indices, page_size);
}

} // namespace infinicore::op
