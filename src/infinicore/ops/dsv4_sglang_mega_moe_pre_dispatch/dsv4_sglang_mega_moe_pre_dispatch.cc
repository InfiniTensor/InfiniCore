#include "infinicore/ops/dsv4_sglang_mega_moe_pre_dispatch.hpp"
#include "../../utils.hpp"
namespace infinicore::op {
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Dsv4SglangMegaMoePreDispatch);
Dsv4SglangMegaMoePreDispatch::Dsv4SglangMegaMoePreDispatch(const Tensor &x, const Tensor &topk_idx, const Tensor &topk_weights, Tensor buf_x, Tensor buf_x_sf, Tensor buf_topk_idx, Tensor buf_topk_weights) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(x, topk_idx, topk_weights, buf_x, buf_x_sf, buf_topk_idx, buf_topk_weights);
    INFINICORE_GRAPH_OP_DISPATCH(x->device().getType(), x, topk_idx, topk_weights, buf_x, buf_x_sf, buf_topk_idx, buf_topk_weights);
}
void Dsv4SglangMegaMoePreDispatch::execute(const Tensor &x, const Tensor &topk_idx, const Tensor &topk_weights, Tensor buf_x, Tensor buf_x_sf, Tensor buf_topk_idx, Tensor buf_topk_weights) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Dsv4SglangMegaMoePreDispatch, x, topk_idx, topk_weights, buf_x, buf_x_sf, buf_topk_idx, buf_topk_weights);
}
void dsv4_sglang_mega_moe_pre_dispatch_(const Tensor &x, const Tensor &topk_idx, const Tensor &topk_weights, Tensor buf_x, Tensor buf_x_sf, Tensor buf_topk_idx, Tensor buf_topk_weights) {
    Dsv4SglangMegaMoePreDispatch::execute(x, topk_idx, topk_weights, buf_x, buf_x_sf, buf_topk_idx, buf_topk_weights);
}
} // namespace infinicore::op
