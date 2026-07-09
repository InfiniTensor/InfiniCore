#include "infinicore/ops/dsv4_sglang_store_flashmla.hpp"
#include "../../utils.hpp"
namespace infinicore::op {
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Dsv4SglangStoreFlashmla);
Dsv4SglangStoreFlashmla::Dsv4SglangStoreFlashmla(const Tensor &input, Tensor cache, const Tensor &indices) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(input, cache, indices);
    INFINICORE_GRAPH_OP_DISPATCH(input->device().getType(), input, cache, indices);
}
void Dsv4SglangStoreFlashmla::execute(const Tensor &input, Tensor cache, const Tensor &indices) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Dsv4SglangStoreFlashmla, input, cache, indices);
}
void dsv4_sglang_store_flashmla_(const Tensor &input, Tensor cache, const Tensor &indices) {
    Dsv4SglangStoreFlashmla::execute(input, cache, indices);
}
} // namespace infinicore::op
