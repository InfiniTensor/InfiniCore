#include "infinicore/ops/mha_kvcache.hpp"
#include "../../utils.hpp"
#include "infinicore/context/context.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(MhaKVCache);

MhaKVCache::MhaKVCache(Tensor out,
                       const Tensor &q,
                       const Tensor &k_cache,
                       const Tensor &v_cache,
                       const Tensor &seqlens_k,
                       const Tensor &block_table,
                       std::optional<Tensor> alibi_slopes,
                       float scale) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, q, k_cache, v_cache, seqlens_k, block_table);
    INFINICORE_GRAPH_OP_DISPATCH(out->device().getType(),
                                 out, q, k_cache, v_cache, seqlens_k, block_table, alibi_slopes, scale);
    // MetaX FA2 mha_fwd_kvcache is not stream-capture-safe by default (HTC under
    // hcStreamBeginCapture). Production keeps FA host_break_; in-graph is
    // FORCE-only via INFINI_FA_FORCE_CAPTURE (faInGraphAllowed is not
    // phase-adaptive). H4: host_break only skips hcStreamBeginCapture — HostOp
    // still runs mha_kvcache_flashattn::run via non-owning to_aten_tensor
    // (from_blob); not a vLLM FX “attn outside torch graph” split.
    host_break_ = !context::faInGraphAllowed();
}

void MhaKVCache::execute(Tensor out,
                         const Tensor &q,
                         const Tensor &k_cache,
                         const Tensor &v_cache,
                         const Tensor &seqlens_k,
                         const Tensor &block_table,
                         std::optional<Tensor> alibi_slopes,
                         float scale) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(
        MhaKVCache,
        out, q, k_cache, v_cache, seqlens_k, block_table, alibi_slopes, scale);
}

void mha_kvcache_(Tensor out,
                  const Tensor &q,
                  const Tensor &k_cache,
                  const Tensor &v_cache,
                  const Tensor &seqlens_k,
                  const Tensor &block_table,
                  std::optional<Tensor> alibi_slopes,
                  float scale) {
    MhaKVCache::execute(out, q, k_cache, v_cache, seqlens_k, block_table, alibi_slopes, scale);
}

Tensor mha_kvcache(const Tensor &q,
                   const Tensor &k_cache,
                   const Tensor &v_cache,
                   const Tensor &seqlens_k,
                   const Tensor &block_table,
                   std::optional<Tensor> alibi_slopes,
                   float scale) {
    // Output shape matches q: [batch_size, seqlen_q, num_heads, head_size]
    auto out = Tensor::empty(q->shape(), q->dtype(), q->device());
    mha_kvcache_(out, q, k_cache, v_cache, seqlens_k, block_table, alibi_slopes, scale);
    return out;
}

} // namespace infinicore::op
