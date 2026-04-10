#pragma once

#include "../device.hpp"
#include "common/op.hpp"
#include <optional>

namespace infinicore::op {

// Graph-recordable InfLLM-v2 attention ops.
//
// These wrappers provide `_` variants that write into a pre-allocated output
// tensor so they can participate in the graph recording system.
INFINICORE_GRAPH_OP_CLASS(
    InfllmV2AttentionVarlen,
    Tensor,
    const Tensor &,
    const Tensor &,
    const Tensor &,
    const Tensor &,
    const Tensor &,
    int,
    int,
    float,
    bool,
    int,
    int);

INFINICORE_GRAPH_OP_CLASS(
    InfllmV2AttentionKVCache,
    Tensor,
    const Tensor &,
    const Tensor &,
    const Tensor &,
    const Tensor &,
    float,
    bool,
    int,
    int);

INFINICORE_GRAPH_OP_CLASS(
    InfllmV2AttentionKVCacheUpdate,
    Tensor,
    const Tensor &,
    const Tensor &,
    const Tensor &,
    const Tensor &,
    const Tensor &,
    const Tensor &,
    float,
    bool,
    int,
    int);

// Varlen InfLLM-V2 attention over unpadded Q/K/V.
//
// Shapes follow the FlashAttn-style varlen convention:
//   q           : [total_q, nheads, head_dim]
//   k, v        : [total_k, nheads_k, head_dim]
//   cu_seqlens_q: [batch_size + 1] (int32)
//   cu_seqlens_k: [batch_size + 1] (int32)
//
// Returns:
//   [total_q, nheads, head_dim]
void infllmv2_varlen_(Tensor out,
                      const Tensor &q,
                      const Tensor &k,
                      const Tensor &v,
                      const Tensor &cu_seqlens_q,
                      const Tensor &cu_seqlens_k,
                      int max_seqlen_q,
                      int max_seqlen_k,
                      float scale,
                      bool causal,
                      int window_size_left = -1,
                      int window_size_right = -1);
Tensor infllmv2_varlen(const Tensor &q,
                       const Tensor &k,
                       const Tensor &v,
                       const Tensor &cu_seqlens_q,
                       const Tensor &cu_seqlens_k,
                       int max_seqlen_q,
                       int max_seqlen_k,
                       float scale,
                       bool causal,
                       int window_size_left = -1,
                       int window_size_right = -1);

// Preferred names (attention-disambiguated). These are header-only aliases to the
// backward-compatible `infllmv2_*` symbols to avoid adding extra exported ABI.
inline void infllmv2_attention_varlen_(Tensor out,
                                       const Tensor &q,
                                       const Tensor &k,
                                       const Tensor &v,
                                       const Tensor &cu_seqlens_q,
                                       const Tensor &cu_seqlens_k,
                                       int max_seqlen_q,
                                       int max_seqlen_k,
                                       float scale,
                                       bool causal,
                                       int window_size_left = -1,
                                       int window_size_right = -1) {
    infllmv2_varlen_(out, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, scale, causal, window_size_left, window_size_right);
}
inline Tensor infllmv2_attention_varlen(const Tensor &q,
                                        const Tensor &k,
                                        const Tensor &v,
                                        const Tensor &cu_seqlens_q,
                                        const Tensor &cu_seqlens_k,
                                        int max_seqlen_q,
                                        int max_seqlen_k,
                                        float scale,
                                        bool causal,
                                        int window_size_left = -1,
                                        int window_size_right = -1) {
    return infllmv2_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, scale, causal, window_size_left, window_size_right);
}

// Decode-time InfLLM-V2 attention with KV cache.
//
// Shapes:
//   q          : [batch, seqlen_q, nheads, head_dim]
//   k_cache    : [num_blocks, block_size, nheads_k, head_dim] or [batch, seqlen_cache, nheads_k, head_dim]
//   v_cache    : same as k_cache
//   cache_lens : [batch] (int32) total KV length per sequence
//
// Returns:
//   [batch, seqlen_q, nheads, head_dim]
void infllmv2_kvcache_(Tensor out,
                       const Tensor &q,
                       const Tensor &k_cache,
                       const Tensor &v_cache,
                       const Tensor &cache_lens,
                       float scale,
                       bool causal,
                       int window_size_left = -1,
                       int window_size_right = -1);
Tensor infllmv2_kvcache(const Tensor &q,
                        const Tensor &k_cache,
                        const Tensor &v_cache,
                        const Tensor &cache_lens,
                        float scale,
                        bool causal,
                        int window_size_left = -1,
                        int window_size_right = -1);

inline void infllmv2_attention_kvcache_(Tensor out,
                                        const Tensor &q,
                                        const Tensor &k_cache,
                                        const Tensor &v_cache,
                                        const Tensor &cache_lens,
                                        float scale,
                                        bool causal,
                                        int window_size_left = -1,
                                        int window_size_right = -1) {
    infllmv2_kvcache_(out, q, k_cache, v_cache, cache_lens, scale, causal, window_size_left, window_size_right);
}
inline Tensor infllmv2_attention_kvcache(const Tensor &q,
                                         const Tensor &k_cache,
                                         const Tensor &v_cache,
                                         const Tensor &cache_lens,
                                         float scale,
                                         bool causal,
                                         int window_size_left = -1,
                                         int window_size_right = -1) {
    return infllmv2_kvcache(q, k_cache, v_cache, cache_lens, scale, causal, window_size_left, window_size_right);
}

// Decode-time InfLLM-V2 attention with KV cache, updating cache in-place.
//
// Shapes:
//   q          : [batch, seqlen_q, nheads, head_dim]
//   k_cache    : [batch, seqlen_cache, nheads_k, head_dim] (dense cache)
//   v_cache    : same as k_cache
//   k_new/v_new: [batch, seqlen_new, nheads_k, head_dim] (new KV to append at cache_lens offsets)
//   cache_lens : [batch] (int32) current KV length per sequence BEFORE appending
//
// Returns:
//   [batch, seqlen_q, nheads, head_dim]
void infllmv2_kvcache_update_(Tensor out,
                              const Tensor &q,
                              const Tensor &k_cache,
                              const Tensor &v_cache,
                              const Tensor &k_new,
                              const Tensor &v_new,
                              const Tensor &cache_lens,
                              float scale,
                              bool causal,
                              int window_size_left = -1,
                              int window_size_right = -1);
Tensor infllmv2_kvcache_update(const Tensor &q,
                               const Tensor &k_cache,
                               const Tensor &v_cache,
                               const Tensor &k_new,
                               const Tensor &v_new,
                               const Tensor &cache_lens,
                               float scale,
                               bool causal,
                               int window_size_left = -1,
                               int window_size_right = -1);

inline void infllmv2_attention_kvcache_update_(Tensor out,
                                               const Tensor &q,
                                               const Tensor &k_cache,
                                               const Tensor &v_cache,
                                               const Tensor &k_new,
                                               const Tensor &v_new,
                                               const Tensor &cache_lens,
                                               float scale,
                                               bool causal,
                                               int window_size_left = -1,
                                               int window_size_right = -1) {
    infllmv2_kvcache_update_(out, q, k_cache, v_cache, k_new, v_new, cache_lens, scale, causal, window_size_left, window_size_right);
}
inline Tensor infllmv2_attention_kvcache_update(const Tensor &q,
                                                const Tensor &k_cache,
                                                const Tensor &v_cache,
                                                const Tensor &k_new,
                                                const Tensor &v_new,
                                                const Tensor &cache_lens,
                                                float scale,
                                                bool causal,
                                                int window_size_left = -1,
                                                int window_size_right = -1) {
    return infllmv2_kvcache_update(q, k_cache, v_cache, k_new, v_new, cache_lens, scale, causal, window_size_left, window_size_right);
}

} // namespace infinicore::op

