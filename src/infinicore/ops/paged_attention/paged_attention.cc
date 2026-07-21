#include "infinicore/ops/paged_attention.hpp"
#include "../../utils.hpp"

#include <cstdlib>
#include <string>

namespace infinicore::op {

namespace {

/// Diagnose: MetaX PagedAttention under hcStream capture probes ATU (binary-search
/// MAX_OPS=10 PASS / 11+ FAIL). Opt-in host-break to keep MoE in-graph while attn
/// stays eager — mirror FA host_break. Default remains in-graph for FORCE path.
bool paged_attn_host_break_enabled() {
    const char *v = std::getenv("INFINI_PAGED_ATTN_HOST_BREAK");
    return v != nullptr && v[0] != '\0' && std::string(v) != "0";
}

} // namespace

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(PagedAttention);

PagedAttention::PagedAttention(Tensor out, const Tensor &q, const Tensor &k_cache, const Tensor &v_cache,
                               const Tensor &block_tables, const Tensor &kv_lens,
                               std::optional<Tensor> alibi_slopes, float scale) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, q, k_cache, v_cache, block_tables, kv_lens);
    INFINICORE_GRAPH_OP_DISPATCH(out->device().getType(),
                                 out, q, k_cache, v_cache, block_tables, kv_lens, alibi_slopes, scale);
    // MetaX: PagedAttention in monolithic decode CG → probe/replay HTC (naive FULL Class B).
    // INFINI_PAGED_ATTN_HOST_BREAK=1 splits graph at attn (PagedCaching can stay in-seg).
    host_break_ = paged_attn_host_break_enabled();
}

void PagedAttention::execute(Tensor out, const Tensor &q, const Tensor &k_cache, const Tensor &v_cache,
                             const Tensor &block_tables, const Tensor &kv_lens,
                             std::optional<Tensor> alibi_slopes, float scale) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(
        PagedAttention,
        out, q, k_cache, v_cache, block_tables, kv_lens, alibi_slopes, scale);
}

Tensor paged_attention(const Tensor &q, const Tensor &k_cache, const Tensor &v_cache,
                       const Tensor &block_tables, const Tensor &kv_lens,
                       std::optional<Tensor> alibi_slopes, float scale) {
    auto out = Tensor::empty(q->shape(), q->dtype(), q->device());
    paged_attention_(out, q, k_cache, v_cache, block_tables, kv_lens, alibi_slopes, scale);
    return out;
}

void paged_attention_(Tensor out, const Tensor &q, const Tensor &k_cache, const Tensor &v_cache,
                      const Tensor &block_tables, const Tensor &kv_lens,
                      std::optional<Tensor> alibi_slopes, float scale) {
    PagedAttention::execute(out, q, k_cache, v_cache, block_tables, kv_lens, alibi_slopes, scale);
}

} // namespace infinicore::op
