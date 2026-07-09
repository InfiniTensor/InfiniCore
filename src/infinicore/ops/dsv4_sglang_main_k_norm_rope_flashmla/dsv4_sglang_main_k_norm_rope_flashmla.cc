#include "infinicore/ops/dsv4_sglang_main_k_norm_rope_flashmla.hpp"
#include "../../utils.hpp"
namespace infinicore::op {
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Dsv4SglangMainKNormRopeFlashmla);
Dsv4SglangMainKNormRopeFlashmla::Dsv4SglangMainKNormRopeFlashmla(Tensor kv, const Tensor &weight, const Tensor &freqs, const Tensor &positions, const Tensor &out_loc, Tensor cache, double eps) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(kv, weight, freqs, positions, out_loc, cache);
    INFINICORE_GRAPH_OP_DISPATCH(kv->device().getType(), kv, weight, freqs, positions, out_loc, cache, eps);
}
void Dsv4SglangMainKNormRopeFlashmla::execute(Tensor kv, const Tensor &weight, const Tensor &freqs, const Tensor &positions, const Tensor &out_loc, Tensor cache, double eps) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Dsv4SglangMainKNormRopeFlashmla, kv, weight, freqs, positions, out_loc, cache, eps);
}
void dsv4_sglang_main_k_norm_rope_flashmla_(Tensor kv, const Tensor &weight, const Tensor &freqs, const Tensor &positions, const Tensor &out_loc, Tensor cache, double eps) {
    Dsv4SglangMainKNormRopeFlashmla::execute(kv, weight, freqs, positions, out_loc, cache, eps);
}
} // namespace infinicore::op
