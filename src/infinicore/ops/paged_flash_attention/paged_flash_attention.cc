#include "infinicore/ops/paged_flash_attention.hpp"

#include "infinicore/ops/mha_kvcache.hpp"
#include "infinicore/ops/mha_varlen.hpp"
#include "infinicore/ops/paged_caching.hpp"

#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace infinicore::op {
namespace {

std::pair<Tensor, Tensor> update_paged_kv_cache(
    const Tensor &key,
    const Tensor &value,
    const Tensor &kv_cache,
    const Tensor &slot_mapping,
    size_t num_heads,
    size_t num_kv_heads,
    size_t head_dim) {
    auto k_cache_layer = kv_cache->narrow({{0, 0, 1}})->squeeze(0);
    auto v_cache_layer = kv_cache->narrow({{0, 1, 1}})->squeeze(0);
    const auto &cache_shape = k_cache_layer->shape();
    const bool use_hygon_lightop_paged_attention =
        key->device().getType() == Device::Type::HYGON
        && cache_shape.size() == 4
        && cache_shape[1] == 64
        && cache_shape[2] == num_kv_heads
        && cache_shape[3] == head_dim
        && num_heads == 8
        && num_kv_heads == 1
        && head_dim == 128;
    if (use_hygon_lightop_paged_attention) {
        const auto num_blocks = cache_shape[0];
        const auto block_size = cache_shape[1];
        auto k_cache_lightop = k_cache_layer->view(
            {num_blocks, num_kv_heads, block_size, head_dim});
        auto v_cache_lightop = v_cache_layer->view(
            {num_blocks, num_kv_heads, head_dim, block_size});
        paged_caching_(
            k_cache_lightop,
            v_cache_lightop,
            key,
            value,
            slot_mapping);
        return {k_cache_lightop, v_cache_lightop};
    }

    paged_caching_(
        k_cache_layer->permute({0, 2, 1, 3}),
        v_cache_layer->permute({0, 2, 1, 3}),
        key,
        value,
        slot_mapping);
    return {k_cache_layer, v_cache_layer};
}

} // namespace

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
    float scale) {
    static std::mutex hygon_paged_flash_attention_mutex;
    std::unique_lock<std::mutex> hygon_lock(
        hygon_paged_flash_attention_mutex,
        std::defer_lock);
    if (query->device().getType() == Device::Type::HYGON) {
        hygon_lock.lock();
    }

    auto [k_total, v_total] = update_paged_kv_cache(
        key,
        value,
        kv_cache,
        slot_mapping,
        num_heads,
        num_kv_heads,
        head_dim);

    const size_t seq_len = query->shape()[0];
    const bool is_prefill = seq_len != total_sequence_lengths->shape()[0];
    auto attn_output = Tensor::empty(
        {seq_len, num_heads, head_dim},
        query->dtype(),
        query->device());

    if (is_prefill) {
        const auto cache_block_size = kv_cache->shape()[2];
        const auto max_cache_seqlen =
            block_tables->shape()[1] * cache_block_size;
        if (seq_len > static_cast<size_t>(std::numeric_limits<int>::max())
            || max_cache_seqlen
                > static_cast<size_t>(std::numeric_limits<int>::max())) {
            throw std::runtime_error(
                "FlashAttention sequence length exceeds int range");
        }
        mha_varlen_(
            attn_output,
            query,
            k_total,
            v_total,
            input_offsets.value(),
            cu_seqlens.value(),
            block_tables,
            static_cast<int>(seq_len),
            static_cast<int>(max_cache_seqlen),
            std::nullopt,
            scale);
    } else {
        auto q_for_fa = query->view({seq_len, 1, num_heads, head_dim});
        auto attn_out_4d = mha_kvcache(
            q_for_fa,
            k_total,
            v_total,
            total_sequence_lengths,
            block_tables,
            std::nullopt,
            scale);
        attn_output =
            attn_out_4d->view({seq_len, num_heads, head_dim});
    }

    return attn_output->view({1, seq_len, num_heads * head_dim});
}

} // namespace infinicore::op
