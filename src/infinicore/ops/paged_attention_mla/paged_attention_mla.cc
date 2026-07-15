#include "infinicore/ops/paged_attention_mla.hpp"

#include "../../utils.hpp"

#include <stdexcept>

#if defined(ENABLE_ATEN) && defined(ENABLE_ILUVATAR_API)
#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/adaptor/vllm_iluvatar_adaptor.hpp"
#endif

namespace infinicore::op {
namespace {

void validate_paged_attention_mla(const Tensor &output,
                                  const Tensor &query,
                                  const Tensor &kv_cache,
                                  const Tensor &block_tables,
                                  const Tensor &context_lens,
                                  int64_t max_context_len) {
    if (!output || !query || !kv_cache || !block_tables || !context_lens) {
        throw std::runtime_error("paged_attention_mla expects non-empty tensors");
    }
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, query, kv_cache, block_tables, context_lens);
    if (query->ndim() != 3 || output->ndim() != 3 || kv_cache->ndim() != 3
        || block_tables->ndim() != 2 || context_lens->ndim() != 1) {
        throw std::runtime_error(
            "paged_attention_mla expects query/output/cache/block_tables/context_lens ranks 3/3/3/2/1");
    }
    if (output->size(0) != query->size(0) || output->size(1) != query->size(1)
        || kv_cache->size(2) != query->size(2) || context_lens->size(0) != query->size(0)
        || block_tables->size(0) != query->size(0)) {
        throw std::runtime_error("paged_attention_mla tensor shapes are inconsistent");
    }
    if (output->dtype() != query->dtype() || kv_cache->dtype() != query->dtype()
        || (query->dtype() != DataType::F16 && query->dtype() != DataType::BF16)) {
        throw std::runtime_error("paged_attention_mla requires matching fp16/bfloat16 data tensors");
    }
    if (context_lens->dtype() != DataType::I32) {
        throw std::runtime_error("paged_attention_mla expects int32 context_lens");
    }
    if (block_tables->dtype() != DataType::I32) {
        throw std::runtime_error("paged_attention_mla expects int32 block_tables for the current Iluvatar SO");
    }
    if (!output->is_contiguous() || !query->is_contiguous() || !kv_cache->is_contiguous()
        || !block_tables->is_contiguous() || !context_lens->is_contiguous()) {
        throw std::runtime_error("paged_attention_mla expects contiguous tensors");
    }
    if (max_context_len <= 0) {
        throw std::runtime_error("paged_attention_mla expects max_context_len > 0");
    }
}

} // namespace

void paged_attention_mla_(Tensor output,
                          const Tensor &query,
                          const Tensor &kv_cache,
                          float scale,
                          const Tensor &block_tables,
                          const Tensor &context_lens,
                          int64_t max_context_len) {
    validate_paged_attention_mla(output, query, kv_cache, block_tables, context_lens, max_context_len);

#if defined(ENABLE_ATEN) && defined(ENABLE_ILUVATAR_API)
    if (output->device().getType() == Device::Type::ILUVATAR) {
        if (!adaptor::vllm_iluvatar::paged_attention_mla_available()) {
            throw std::runtime_error("paged_attention_mla requires vllm_iluvatar cuinfer extension");
        }
        auto output_at = adaptor::to_aten_tensor(output);
        auto query_at = adaptor::to_aten_tensor(query);
        auto kv_cache_at = adaptor::to_aten_tensor(kv_cache);
        auto block_tables_at = adaptor::to_aten_tensor(block_tables);
        auto context_lens_at = adaptor::to_aten_tensor(context_lens);
        auto softmax_lse = Tensor::empty(
            {query->size(0), query->size(1)}, DataType::F32, query->device());
        auto softmax_lse_at = adaptor::to_aten_tensor(softmax_lse);
        adaptor::vllm_iluvatar::paged_attention_mla(
            output_at, query_at, kv_cache_at, scale, block_tables_at, context_lens_at,
            max_context_len, false, softmax_lse_at);
        return;
    }
#endif

    throw std::runtime_error("paged_attention_mla currently supports only Iluvatar builds with ATen");
}

} // namespace infinicore::op
