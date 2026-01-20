#include "infinicore/ops/flash_attention.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(FlashAttention);

FlashAttention::FlashAttention(Tensor out, Tensor q, Tensor k, Tensor v, std::size_t total_kv_len, float scale, bool is_causal) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, q, k, v);
    INFINICORE_GRAPH_OP_DISPATCH(out->device().getType(),
                                 out, q, k, v, total_kv_len, scale, is_causal);
}

void FlashAttention::execute(Tensor out, Tensor q, Tensor k, Tensor v, std::size_t total_kv_len, float scale, bool is_causal) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(FlashAttention, out, q, k, v, total_kv_len, scale, is_causal);
}

Tensor flash_attention(Tensor q, Tensor k, Tensor v, std::size_t total_kv_len, float scale, bool is_causal) {
    Shape shape = q->shape();
    auto out = Tensor::empty(shape, q->dtype(), q->device());
    flash_attention_(out, q, k, v, total_kv_len, scale, is_causal);
    return out;
}

void flash_attention_(Tensor out, Tensor q, Tensor k, Tensor v, std::size_t total_kv_len, float scale, bool is_causal) {
    FlashAttention::execute(out, q, k, v, total_kv_len, scale, is_causal);
}
} // namespace infinicore::op
