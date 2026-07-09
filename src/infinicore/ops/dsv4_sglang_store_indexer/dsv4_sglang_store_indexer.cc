#include "infinicore/ops/dsv4_sglang_store_indexer.hpp"
#include "../../utils.hpp"
namespace infinicore::op {
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Dsv4SglangStoreIndexer);
Dsv4SglangStoreIndexer::Dsv4SglangStoreIndexer(const Tensor &input, Tensor cache, const Tensor &indices) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(input, cache, indices);
    INFINICORE_GRAPH_OP_DISPATCH(input->device().getType(), input, cache, indices);
}
void Dsv4SglangStoreIndexer::execute(const Tensor &input, Tensor cache, const Tensor &indices) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Dsv4SglangStoreIndexer, input, cache, indices);
}
void dsv4_sglang_store_indexer_(const Tensor &input, Tensor cache, const Tensor &indices) {
    Dsv4SglangStoreIndexer::execute(input, cache, indices);
}
} // namespace infinicore::op
