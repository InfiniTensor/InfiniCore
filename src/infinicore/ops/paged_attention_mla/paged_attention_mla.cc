#include "infinicore/ops/paged_attention_mla.hpp"

#include "../../utils.hpp"

#include <stdexcept>

#include "../vendor_ops/vendor_ops_dispatch.hpp"

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
    if (block_tables->dtype() != DataType::I32 && block_tables->dtype() != DataType::I64) {
        throw std::runtime_error("paged_attention_mla expects int32 or int64 block_tables");
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

    auto kernel = vendor_ops::lookup(
        vendor_ops::paged_attention_mla_dispatcher(),
        output->device().getType(),
        "paged_attention_mla");
    kernel(output, query, kv_cache, scale, block_tables, context_lens, max_context_len);
}

} // namespace infinicore::op
