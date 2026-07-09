#include "infinicore/ops/dsv4_sglang_paged_mqa_logits_metadata.hpp"

#include "../../utils.hpp"
namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Dsv4SglangPagedMqaLogitsMetadata);

Dsv4SglangPagedMqaLogitsMetadata::Dsv4SglangPagedMqaLogitsMetadata(const Tensor &seq_lens, Tensor metadata) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(seq_lens, metadata);
    INFINICORE_GRAPH_OP_DISPATCH(seq_lens->device().getType(), seq_lens, metadata);
}

void Dsv4SglangPagedMqaLogitsMetadata::execute(const Tensor &seq_lens, Tensor metadata) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Dsv4SglangPagedMqaLogitsMetadata, seq_lens, metadata);
}

void dsv4_sglang_paged_mqa_logits_metadata_(const Tensor &seq_lens, Tensor metadata) {
    Dsv4SglangPagedMqaLogitsMetadata::execute(seq_lens, metadata);
}

} // namespace infinicore::op
