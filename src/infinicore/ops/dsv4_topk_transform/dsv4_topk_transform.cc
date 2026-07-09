#include "infinicore/ops/dsv4_topk_transform.hpp"

#include "../../utils.hpp"
namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Dsv4TopkTransform);

Dsv4TopkTransform::Dsv4TopkTransform(Tensor out, const Tensor &scores, const Tensor &seq_lens, const Tensor &page_tables, int page_size) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, scores, seq_lens, page_tables);
    INFINICORE_GRAPH_OP_DISPATCH(out->device().getType(), out, scores, seq_lens, page_tables, page_size);
}

void Dsv4TopkTransform::execute(Tensor out, const Tensor &scores, const Tensor &seq_lens, const Tensor &page_tables, int page_size) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Dsv4TopkTransform, out, scores, seq_lens, page_tables, page_size);
}

Tensor dsv4_topk_transform(const Tensor &scores, const Tensor &seq_lens, const Tensor &page_tables, int page_size) {
    auto shape = scores->shape();
    INFINICORE_ASSERT(shape.size() == 2 && shape[1] % 64 == 0);
    auto out = Tensor::empty({shape[0], shape[1] / 64}, DataType::I32, scores->device());
    dsv4_topk_transform_(out, scores, seq_lens, page_tables, page_size);
    return out;
}

void dsv4_topk_transform_(Tensor out, const Tensor &scores, const Tensor &seq_lens, const Tensor &page_tables, int page_size) {
    Dsv4TopkTransform::execute(out, scores, seq_lens, page_tables, page_size);
}

} // namespace infinicore::op
