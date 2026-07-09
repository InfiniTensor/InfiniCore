#include "infinicore/ops/dsv4_swa_prefill_indices.hpp"
namespace infinicore::op {
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Dsv4SwaPrefillIndices);
Dsv4SwaPrefillIndices::Dsv4SwaPrefillIndices(Tensor indices, int window_size) {
    INFINICORE_GRAPH_OP_DISPATCH(indices->device().getType(), indices, window_size);
}
void Dsv4SwaPrefillIndices::execute(Tensor indices, int window_size) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Dsv4SwaPrefillIndices, indices, window_size);
}
void dsv4_swa_prefill_indices_(Tensor indices, int window_size) { Dsv4SwaPrefillIndices::execute(indices, window_size); }
} // namespace infinicore::op
