#pragma once

#include "../tensor.hpp"

#include <cstddef>
#include <optional>

namespace infinicore::op {

// Update a paged KV cache and run prefill or decode FlashAttention.
//
// On Hygon, this function also owns the LightOP cache-layout policy and
// serializes graph capture across TP threads because the vendor extension
// keeps process-global launch state.
Tensor paged_flash_attention(
    const Tensor &query,
    const Tensor &key,
    const Tensor &value,
    const Tensor &kv_cache,
    const Tensor &total_sequence_lengths,
    const std::optional<Tensor> &input_offsets,
    const std::optional<Tensor> &cu_seqlens,
    const Tensor &block_tables,
    const Tensor &slot_mapping,
    size_t num_heads,
    size_t num_kv_heads,
    size_t head_dim,
    float scale);

} // namespace infinicore::op
