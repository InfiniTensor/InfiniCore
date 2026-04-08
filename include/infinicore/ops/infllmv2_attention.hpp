#pragma once

#include "../device.hpp"
#include "common/op.hpp"
#include <optional>

namespace infinicore::op {

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
Tensor infllmv2_kvcache(const Tensor &q,
                        const Tensor &k_cache,
                        const Tensor &v_cache,
                        const Tensor &cache_lens,
                        float scale,
                        bool causal,
                        int window_size_left = -1,
                        int window_size_right = -1);

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

} // namespace infinicore::op

