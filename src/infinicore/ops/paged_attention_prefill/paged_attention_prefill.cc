#include "infinicore/ops/paged_attention_prefill.hpp"

#include "../../utils.hpp"

#if defined(ENABLE_ILUVATAR_FLASH_ATTN)

#include "infinicore/adaptor/aten_adaptor.hpp"

#include <ATen/ATen.h>
#include <optional>
#include <stdexcept>
namespace ixformer {
namespace infer {
void ixinfer_flash_attn_unpad_with_block_tables(
    at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &,
    at::Tensor &, at::Tensor &, long, long, bool, long, long,
    long, double, double, bool, const std::optional<at::Tensor> &,
    const std::optional<at::Tensor> &, std::optional<at::Tensor> &);
} // namespace infer
} // namespace ixformer
#endif
namespace infinicore::op {

common::OpDispatcher<PagedAttentionPrefill::schema> &PagedAttentionPrefill::dispatcher() {
    static common::OpDispatcher<PagedAttentionPrefill::schema> dispatcher_;
    return dispatcher_;
};

void PagedAttentionPrefill::execute(Tensor out, Tensor q, Tensor k_cache, Tensor v_cache,
                                    Tensor block_tables, Tensor kv_lens, Tensor cum_seqlens_q,
                                    std::optional<Tensor> alibi_slopes, float scale) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, q, k_cache, v_cache, block_tables, kv_lens, cum_seqlens_q);

    infinicore::context::setDevice(out->device());

    dispatcher().lookup(out->device().getType())(out, q, k_cache, v_cache, block_tables,
                                                 kv_lens, cum_seqlens_q, alibi_slopes, scale);
}

Tensor paged_attention_prefill(Tensor q, Tensor k_cache, Tensor v_cache,
                               Tensor block_tables, Tensor kv_lens, Tensor cum_seqlens_q,
                               std::optional<Tensor> alibi_slopes, float scale) {

    auto out = Tensor::empty(q->shape(), q->dtype(), q->device());
    paged_attention_prefill_(out, q, k_cache, v_cache, block_tables, kv_lens, cum_seqlens_q, alibi_slopes, scale);
    return out;
}

void paged_attention_prefill_(Tensor out, Tensor q, Tensor k_cache, Tensor v_cache,
                              Tensor block_tables, Tensor kv_lens, Tensor cum_seqlens_q,
                              std::optional<Tensor> alibi_slopes, float scale) {

    PagedAttentionPrefill::execute(out, q, k_cache, v_cache, block_tables, kv_lens, cum_seqlens_q, alibi_slopes, scale);
}

void paged_attention_prefill_mtp_(Tensor out, Tensor q, Tensor k_cache, Tensor v_cache,
                                  Tensor block_tables, Tensor total_kv_lens, Tensor cum_seqlens_q,
                                  std::optional<Tensor> alibi_slopes, float scale,
                                  size_t max_query_len,
                                  Tensor cu_seqlens_k,
                                  size_t max_kv_len) {
#if defined(ENABLE_ILUVATAR_FLASH_ATTN)

    if (alibi_slopes.has_value()) {
        throw std::runtime_error(
            "Iluvatar prefill flash attention does not support alibi_slopes");
    }
    if (q->ndim() != 3 || out->shape() != q->shape() || !q->is_contiguous() ||
        !out->is_contiguous()) {
        throw std::runtime_error(
            "Iluvatar prefill flash attention expects packed contiguous q/out");
    }
    if (k_cache->ndim() != 4 || v_cache->shape() != k_cache->shape() ||

        k_cache->size(1) == 0 || k_cache->size(2) == 0 ||
        q->size(1) % k_cache->size(1) != 0 ||
        q->size(2) != k_cache->size(3)) {
        throw std::runtime_error(
            "Iluvatar prefill flash attention expects BHSD kv caches with GQA heads");
    }
    if (block_tables->ndim() != 2 || block_tables->dtype() != DataType::I32 ||
        !block_tables->is_contiguous() || block_tables->size(0) != total_kv_lens->size(0) ||
        total_kv_lens->dtype() != DataType::I32 || !total_kv_lens->is_contiguous() ||
        cum_seqlens_q->dtype() != DataType::I32 || !cum_seqlens_q->is_contiguous() ||
        cum_seqlens_q->size(0) != total_kv_lens->size(0) + 1 ||
        cu_seqlens_k->dtype() != DataType::I32 || !cu_seqlens_k->is_contiguous() ||
        cu_seqlens_k->size(0) != total_kv_lens->size(0) + 1 || max_query_len == 0 ||
        max_kv_len == 0) {
        throw std::runtime_error("Invalid Iluvatar prefill flash attention metadata");
    }

    auto output_tensor = adaptor::to_aten_tensor(out);
    auto query_tensor = adaptor::to_aten_tensor(q);
    auto key_tensor = adaptor::to_aten_tensor(k_cache);
    auto value_tensor = adaptor::to_aten_tensor(v_cache);
    auto block_table_tensor = adaptor::to_aten_tensor(block_tables);
    auto cu_query_lens_tensor = adaptor::to_aten_tensor(cum_seqlens_q);
    auto cu_kv_lens_tensor = adaptor::to_aten_tensor(cu_seqlens_k);
    std::optional<at::Tensor> softmax_lse;

    adaptor::set_aten_stream_to_infinicore();
    ::ixformer::infer::ixinfer_flash_attn_unpad_with_block_tables(
        query_tensor,
        key_tensor,
        value_tensor,
        output_tensor,
        block_table_tensor,
        cu_query_lens_tensor,
        cu_kv_lens_tensor,
        static_cast<long>(max_query_len),
        static_cast<long>(max_kv_len),
        true,
        -1,
        -1,
        0,
        scale,
        0.0,
        false,
        std::nullopt,
        std::nullopt,
        softmax_lse);
#else
    throw std::runtime_error("Iluvatar prefill flash attention is not enabled");
#endif
}
} // namespace infinicore::op
