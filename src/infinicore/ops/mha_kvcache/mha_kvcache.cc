#include "infinicore/ops/mha_kvcache.hpp"
#include "../../utils.hpp"

#include <cstdlib>
#include <string>

namespace infinicore::op {

namespace {

/// Diagnose-only (P8a): force FA into hcStreamBeginCapture. Default off — MetaX FA2
/// historically HTC / IllegalAddress under capture. Never set in production serve.
bool fa_force_capture_enabled() {
    const char *v = std::getenv("INFINI_FA_FORCE_CAPTURE");
    return v != nullptr && v[0] != '\0' && std::string(v) != "0";
}

} // namespace

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
    // MetaX FA2 mha_fwd_kvcache is not stream-capture-safe (HTC mem violation under
    // hcStreamBeginCapture). Mirror MoE / prefill FA2 piecewise: eager between device segments.
    // Opt-in INFINI_FA_FORCE_CAPTURE=1 for fa_capture_smoke diagnose only.
    host_break_ = !fa_force_capture_enabled();
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
