#if defined(ENABLE_CAMBRICON_API) && defined(ENABLE_FLASH_ATTN)

#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/adaptor/cambricon_flash_attention.hpp"
#include "infinicore/ops/mha_varlen.hpp"

#include <ATen/ops/scaled_dot_product_attention.h>

#include <stdexcept>

namespace infinicore::op::mha_varlen_impl::cambricon {

using adaptor::cambricon_flash_attn::gather_paged_cache;
using adaptor::cambricon_flash_attn::lengths_from_cumulative;
using adaptor::cambricon_flash_attn::to_host_i32;

struct PlannedMeta {
    graph::GraphTensor out, q, k, v, cum_seqlens_q, cum_seqlens_k;
    std::optional<graph::GraphTensor> block_table;
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
           std::optional<Tensor> block_table,
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
        block_table
            ? std::optional<graph::GraphTensor>(
                graph::GraphTensor(*block_table))
            : std::nullopt,
        max_seqlen_q,
        max_seqlen_k,
        alibi_slopes
            ? std::optional<graph::GraphTensor>(graph::GraphTensor(*alibi_slopes))
            : std::nullopt,
        scale};
}

namespace {

void run_sdpa_fallback(const at::Tensor &q,
                       const at::Tensor &k,
                       const at::Tensor &v,
                       at::Tensor &out,
                       const at::Tensor &cu_q,
                       const at::Tensor &cu_k,
                       float scale) {
    const auto q_offsets = to_host_i32(cu_q);
    const auto k_offsets = to_host_i32(cu_k);
    if (q_offsets.size() != k_offsets.size()) {
        throw std::runtime_error(
            "Cambricon MHA varlen has mismatched cumulative lengths");
    }
    for (size_t batch = 0; batch + 1 < q_offsets.size(); ++batch) {
        const int64_t q_begin = q_offsets[batch];
        const int64_t q_end = q_offsets[batch + 1];
        const int64_t k_begin = k_offsets[batch];
        const int64_t k_end = k_offsets[batch + 1];
        auto q_item = q.slice(0, q_begin, q_end).transpose(0, 1).unsqueeze(0);
        auto k_item = k.slice(0, k_begin, k_end).transpose(0, 1).unsqueeze(0);
        auto v_item = v.slice(0, k_begin, k_end).transpose(0, 1).unsqueeze(0);
        if (q_item.size(1) % k_item.size(1) != 0) {
            throw std::runtime_error(
                "Cambricon MHA varlen query heads must be divisible by KV heads");
        }
        const int64_t groups = q_item.size(1) / k_item.size(1);
        if (groups > 1) {
            k_item = k_item.repeat_interleave(groups, 1);
            v_item = v_item.repeat_interleave(groups, 1);
        }
        auto result = at::scaled_dot_product_attention(
            q_item, k_item, v_item, std::nullopt, 0.0, true,
            std::optional<double>(static_cast<double>(scale)));
        out.slice(0, q_begin, q_end)
            .copy_(result.squeeze(0).transpose(0, 1));
    }
}

} // namespace

void run(void *planned_meta) {
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);
    torch_mlu::mlu::MLUStreamGuard guard(
        infinicore::adaptor::get_mlu_stream());
    auto q = infinicore::adaptor::to_aten_tensor(p->q).contiguous();
    auto k = infinicore::adaptor::to_aten_tensor(p->k);
    auto v = infinicore::adaptor::to_aten_tensor(p->v);
    auto cu_q = infinicore::adaptor::to_aten_tensor(p->cum_seqlens_q);
    auto cu_k = infinicore::adaptor::to_aten_tensor(p->cum_seqlens_k);

    const bool copy_back = !p->out->is_contiguous();
    Tensor out_work_ic
        = copy_back ? p->out->contiguous() : Tensor(p->out);
    auto out_tensor
        = infinicore::adaptor::to_aten_tensor(out_work_ic);

    int max_q = 0;
    for (int length : lengths_from_cumulative(cu_q)) {
        max_q = std::max(max_q, length);
    }
    int max_k = 0;
    if (p->block_table) {
        auto block_table
            = infinicore::adaptor::to_aten_tensor(*p->block_table);
        auto packed = gather_paged_cache(
            k, v, block_table, lengths_from_cumulative(cu_k));
        k = std::move(packed.key);
        v = std::move(packed.value);
        max_k = packed.max_seqlen;
    } else {
        k = k.contiguous();
        v = v.contiguous();
        for (int length : lengths_from_cumulative(cu_k)) {
            max_k = std::max(max_k, length);
        }
    }

    const bool flash_supported
        = q.size(-1) <= 512 && k.size(-1) == q.size(-1)
       && v.size(-1) == q.size(-1);
    if (!flash_supported) {
        if (p->alibi_slopes) {
            throw std::runtime_error(
                "Cambricon MHA varlen SDPA fallback does not support ALiBi");
        }
        run_sdpa_fallback(
            q, k, v, out_tensor, cu_q, cu_k, p->scale);
    } else {
        auto out = std::optional<at::Tensor>(out_tensor);
        std::optional<at::Tensor> seqused_k = std::nullopt;
        auto alibi = p->alibi_slopes
                       ? std::optional<at::Tensor>(
                           infinicore::adaptor::to_aten_tensor(*p->alibi_slopes))
                       : std::nullopt;
        ::mha_varlen_fwd(
            q, k, v, out, cu_q, cu_k, seqused_k, alibi,
            max_q, max_k, 0.0F, p->scale, false, true,
            -1, -1, false, std::nullopt);
    }
    if (copy_back) {
        p->out->copy_from(out_work_ic);
    }
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

static bool registered = []() {
    MultiheadAttentionVarlen::plan_dispatcher().registerDevice(
        Device::Type::CAMBRICON, &plan);
    MultiheadAttentionVarlen::run_dispatcher().registerDevice(
        Device::Type::CAMBRICON, &run);
    MultiheadAttentionVarlen::cleanup_dispatcher().registerDevice(
        Device::Type::CAMBRICON, &cleanup);
    return true;
}();

} // namespace infinicore::op::mha_varlen_impl::cambricon

#endif
