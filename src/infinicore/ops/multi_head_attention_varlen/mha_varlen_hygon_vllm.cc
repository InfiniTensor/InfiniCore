// Hygon DCU prefill attention backend using flash::vllm_mha_varlen_fwd.
//
// Replaces the generic mha_varlen_fwd with the vLLM variant which is
// optimized for paged KV-cache prefill on Hygon DCU.

#if defined(ENABLE_FLASH_ATTN) && defined(ENABLE_HYGON_API) && !defined(ENABLE_NVIDIA_API)

#include "infinicore/ops/mha_varlen.hpp"

#include "infinicore/adaptor/flash_attention_adaptor.hpp"

#include <stdexcept>

namespace infinicore::op::mha_varlen_impl::hygon_vllm {

struct PlannedMeta {
    graph::GraphTensor out, q, k, v, cum_seqlens_q, cum_seqlens_k, block_table;
    int max_seqlen_q, max_seqlen_k;
    std::optional<graph::GraphTensor> alibi_slopes;
    float scale;
};

void *plan(Tensor out,
           const Tensor &q,
           const Tensor &k,
           const Tensor &v,
           const Tensor &cum_seqlens_q,
           const Tensor &cum_seqlens_k,
           const Tensor &block_table,
           int max_seqlen_q,
           int max_seqlen_k,
           std::optional<Tensor> alibi_slopes,
           float scale) {
    return new PlannedMeta{
        graph::GraphTensor(out),
        graph::GraphTensor(q),
        graph::GraphTensor(k),
        graph::GraphTensor(v),
        graph::GraphTensor(cum_seqlens_q),
        graph::GraphTensor(cum_seqlens_k),
        graph::GraphTensor(block_table),
        max_seqlen_q,
        max_seqlen_k,
        alibi_slopes ? std::optional<graph::GraphTensor>(graph::GraphTensor(*alibi_slopes)) : std::nullopt,
        scale};
}

void run(void *planned_meta) {
    infinicore::adaptor::TorchStreamGuard guard(infinicore::adaptor::get_cuda_stream());
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);

    auto q = infinicore::adaptor::to_aten_tensor(p->q);
    // InfiniLM passes K/V in permuted layout [num_blocks, block_size, nkv, hd].
    // vllm_mha_varlen_fwd expects:
    //   K: [num_blocks, nkv, block_size, hd]        — standard cache layout
    //   V: [num_blocks, nkv, hd, block_size]         — transposed V layout
    // K: permute back {0,2,1,3}; double-permute restores contiguous strides (no-op copy).
    auto k = infinicore::adaptor::to_aten_tensor(p->k).permute({0, 2, 1, 3}).contiguous();
    // V: from [num_blocks, block_size, nkv, hd] → [num_blocks, nkv, hd, block_size]
    auto v = infinicore::adaptor::to_aten_tensor(p->v).permute({0, 2, 3, 1}).contiguous();
    auto out = std::optional<at::Tensor>(infinicore::adaptor::to_aten_tensor(p->out));
    auto cu_seqlens_q = infinicore::adaptor::to_aten_tensor(p->cum_seqlens_q);
    auto cu_seqlens_kv = infinicore::adaptor::to_aten_tensor(p->cum_seqlens_k);
    auto block_table = std::optional<at::Tensor>(infinicore::adaptor::to_aten_tensor(p->block_table));

    auto device = q.device();
    if (!cu_seqlens_q.is_cuda()) cu_seqlens_q = cu_seqlens_q.to(device);
    if (!cu_seqlens_kv.is_cuda()) cu_seqlens_kv = cu_seqlens_kv.to(device);
    if (block_table.has_value() && !block_table->is_cuda()) block_table = block_table->to(device);

    std::optional<at::Tensor> seqused_k = std::nullopt;
    std::optional<const at::Tensor> leftpad_k = std::nullopt;
    auto alibi_slopes = p->alibi_slopes
                          ? std::optional<at::Tensor>(infinicore::adaptor::to_aten_tensor(*p->alibi_slopes))
                          : std::nullopt;

    flash::vllm_mha_varlen_fwd(
        q, k, v, out,
        cu_seqlens_q, cu_seqlens_kv,
        seqused_k, leftpad_k, block_table, alibi_slopes,
        p->max_seqlen_q, p->max_seqlen_k,
        0.0f, p->scale, false, true,
        -1, -1, 0.0f, false,
        std::nullopt);
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

// Register for Hygon device only, overriding the ALLDEVICE flashattn registration.
static bool registered = []() {
    MultiheadAttentionVarlen::plan_dispatcher().registerDevice(Device::Type::HYGON, &plan, true);
    MultiheadAttentionVarlen::run_dispatcher().registerDevice(Device::Type::HYGON, &run, true);
    MultiheadAttentionVarlen::cleanup_dispatcher().registerDevice(Device::Type::HYGON, &cleanup, true);
    return true;
}();

} // namespace infinicore::op::mha_varlen_impl::hygon_vllm

#endif // ENABLE_FLASH_ATTN && ENABLE_HYGON_API && !ENABLE_NVIDIA_API
