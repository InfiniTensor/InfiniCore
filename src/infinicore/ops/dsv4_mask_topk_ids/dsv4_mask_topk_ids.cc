#include "infinicore/ops/dsv4_mask_topk_ids.hpp"

#include "../../utils.hpp"
namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Dsv4MaskTopkIds);

Dsv4MaskTopkIds::Dsv4MaskTopkIds(Tensor topk_ids, const Tensor &num_token_non_padded) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(topk_ids, num_token_non_padded);
    INFINICORE_GRAPH_OP_DISPATCH(topk_ids->device().getType(), topk_ids, num_token_non_padded);
}

void Dsv4MaskTopkIds::execute(Tensor topk_ids, const Tensor &num_token_non_padded) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Dsv4MaskTopkIds, topk_ids, num_token_non_padded);
}

void dsv4_mask_topk_ids_(Tensor topk_ids, const Tensor &num_token_non_padded) {
    Dsv4MaskTopkIds::execute(topk_ids, num_token_non_padded);
}

} // namespace infinicore::op
