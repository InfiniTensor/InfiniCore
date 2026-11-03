#include "infinicore/ops/mha_kvcache.hpp"
#include "../../utils.hpp"
#include <cstdlib>
#include <cstring>

namespace infinicore::op {

common::OpDispatcher<graph::graph_replay_schema> &
mha_kvcache_graph_replay_dispatcher() {
    static common::OpDispatcher<graph::graph_replay_schema> dispatcher;
    return dispatcher;
}

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
    const char *async_flag = std::getenv("INFINICORE_ASCEND_ASYNC_GRAPH_UPDATE");
    const bool launch_first = out->device().getType() == Device::Type::ASCEND
        && async_flag && std::strcmp(async_flag, "1") == 0;
    set_graph_replay_handler(
        mha_kvcache_graph_replay_dispatcher().lookup(out->device().getType()),
        launch_first);
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
