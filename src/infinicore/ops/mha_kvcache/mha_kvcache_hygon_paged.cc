// Hygon DCU decode attention backend using flash::paged_attention.
//
// Replaces mha_fwd_kvcache (which calls torch::empty for softmax_lse,
// breaking HIP graph capture) with paged_attention (static allocations only).
//
// paged_attention requires V in transposed layout [num_blocks, nkv, hd, block_size].
// We pre-allocate the transposed buffer in plan() and do an ATen copy in run()
// from the standard V cache [num_blocks, nkv, block_size, hd].

#if defined(ENABLE_FLASH_ATTN) && defined(ENABLE_HYGON_API) && !defined(ENABLE_NVIDIA_API)

#include "infinicore/ops/mha_kvcache.hpp"

#include "infinicore/adaptor/flash_attention_adaptor.hpp"

#include <stdexcept>

namespace infinicore::op::mha_kvcache_impl::hygon_paged {

struct PlannedMeta {
    graph::GraphTensor out, q, k_cache, v_cache, seqlens_k, block_table;
    std::optional<graph::GraphTensor> alibi_slopes;
    float scale;
    // Pre-allocated buffer for transposed V: [num_blocks, nkv, hd, block_size]
    graph::GraphTensor v_transposed_buf;
};

void *plan(Tensor out,
           const Tensor &q,
           const Tensor &k_cache,
           const Tensor &v_cache,
           const Tensor &seqlens_k,
           const Tensor &block_table,
           std::optional<Tensor> alibi_slopes,
           float scale) {
    // v_cache arrives permuted: [num_blocks, block_size, nkv, hd]
    // paged_attention needs V transposed: [num_blocks, nkv, hd, block_size]
    auto vs = v_cache->shape(); // {num_blocks, block_size, nkv, hd}
    auto v_transposed = Tensor::empty(
        {vs[0], vs[2], vs[3], vs[1]}, // {num_blocks, nkv, hd, block_size}
        v_cache->dtype(),
        v_cache->device());

    return new PlannedMeta{
        graph::GraphTensor(out),
        graph::GraphTensor(q),
        graph::GraphTensor(k_cache),
        graph::GraphTensor(v_cache),
        graph::GraphTensor(seqlens_k),
        graph::GraphTensor(block_table),
        alibi_slopes ? std::optional<graph::GraphTensor>(graph::GraphTensor(*alibi_slopes)) : std::nullopt,
        scale,
        graph::GraphTensor(v_transposed)};
}

void run(void *planned_meta) {
    infinicore::adaptor::TorchStreamGuard guard(infinicore::adaptor::get_cuda_stream());
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);

    auto out_t = infinicore::adaptor::to_aten_tensor(p->out);
    auto q_t = infinicore::adaptor::to_aten_tensor(p->q);
    // InfiniLM passes K/V in permuted layout [num_blocks, block_size, nkv, hd].
    // paged_attention expects K: [num_blocks, nkv, block_size, hd]
    // Permute back; double-permute restores original contiguous strides (no-op copy).
    auto k_t = infinicore::adaptor::to_aten_tensor(p->k_cache).permute({0, 2, 1, 3}).contiguous();
    auto v_raw = infinicore::adaptor::to_aten_tensor(p->v_cache);
    auto v_transposed = infinicore::adaptor::to_aten_tensor(p->v_transposed_buf);
    auto seqlens = infinicore::adaptor::to_aten_tensor(p->seqlens_k);
    auto block_tbl = infinicore::adaptor::to_aten_tensor(p->block_table);

    // Transpose V on GPU:
    //   v_raw: [num_blocks, block_size, nkv, hd] (permuted from InfiniLM)
    //   target: [num_blocks, nkv, hd, block_size]
    v_transposed.copy_(v_raw.permute({0, 2, 3, 1}));

    // Both q and out are 4D: [num_seqs, seqlen, num_heads, head_dim]
    // paged_attention accesses query.size(2) for num_heads and query.size(3) for head_size,
    // so query must be 4D (despite misleading comment in the source).

    // Compute a safe upper bound for max_context_len from tensor shapes.
    // Using seqlens.max().item<int>() would require a D2H transfer that breaks
    // HIP graph capture. paged_attention uses per-sequence seqlens internally,
    // so a larger bound only wastes a few grid blocks.
    int block_size = static_cast<int>(k_t.size(2));  // after permute-back: [num_blocks, nkv, block_size, hd]
    int max_blocks_per_seq = static_cast<int>(block_tbl.size(1));
    int max_context_len = max_blocks_per_seq * block_size;

    std::optional<at::Tensor> alibi = std::nullopt;
    std::optional<at::Tensor> q_scale = std::nullopt;
    std::optional<at::Tensor> k_scale = std::nullopt;
    std::optional<at::Tensor> v_scale = std::nullopt;

    flash::paged_attention(
        out_t,            // out: [num_seqs, 1, num_heads, head_dim] (4D)
        q_t,              // query: [num_seqs, 1, num_heads, head_dim] (4D)
        k_t,              // [num_blocks, nkv, block_size, hd]
        v_transposed,     // [num_blocks, nkv, hd, block_size]
        p->scale,
        block_tbl,
        seqlens,
        alibi,
        std::string("auto"),
        q_scale, k_scale, v_scale,
        max_context_len);
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

// Register for Hygon device only, overriding the ALLDEVICE flashattn registration.
static bool registered = []() {
    MhaKVCache::plan_dispatcher().registerDevice(Device::Type::HYGON, &plan, true);
    MhaKVCache::run_dispatcher().registerDevice(Device::Type::HYGON, &run, true);
    MhaKVCache::cleanup_dispatcher().registerDevice(Device::Type::HYGON, &cleanup, true);
    return true;
}();

} // namespace infinicore::op::mha_kvcache_impl::hygon_paged

#endif // ENABLE_FLASH_ATTN && ENABLE_HYGON_API && !ENABLE_NVIDIA_API
