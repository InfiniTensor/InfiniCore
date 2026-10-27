#include "infinicore/ops/flash_mla.hpp"

#include "../../utils.hpp"

#include <stdexcept>

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(FlashMlaSparsePrefill);
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(FlashMlaSparseDecode);
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(FlashMlaDenseDecode);

FlashMlaSparsePrefill::FlashMlaSparsePrefill(Tensor output, const Tensor &query,
                                             const Tensor &kv_cache,
                                             const Tensor &indices,
                                             const Tensor &topk_lens,
                                             float scale) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, query, kv_cache, indices,
                                          topk_lens);
    INFINICORE_GRAPH_OP_DISPATCH(output->device().getType(), output, query,
                                 kv_cache, indices, topk_lens, scale);
}

void FlashMlaSparsePrefill::execute(Tensor output, const Tensor &query,
                                    const Tensor &kv_cache,
                                    const Tensor &indices,
                                    const Tensor &topk_lens, float scale) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(FlashMlaSparsePrefill, output, query,
                                      kv_cache, indices, topk_lens, scale);
}

FlashMlaSparseDecode::FlashMlaSparseDecode(Tensor output, const Tensor &query,
                                           const Tensor &kv_cache,
                                           const Tensor &indices,
                                           const Tensor &topk_lens,
                                           float scale) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, query, kv_cache, indices,
                                          topk_lens);
    INFINICORE_GRAPH_OP_DISPATCH(output->device().getType(), output, query,
                                 kv_cache, indices, topk_lens, scale);
}

void FlashMlaSparseDecode::execute(Tensor output, const Tensor &query,
                                   const Tensor &kv_cache,
                                   const Tensor &indices,
                                   const Tensor &topk_lens, float scale) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(FlashMlaSparseDecode, output, query,
                                      kv_cache, indices, topk_lens, scale);
}

FlashMlaDenseDecode::FlashMlaDenseDecode(Tensor output, const Tensor &query,
                                         const Tensor &kv_cache,
                                         const Tensor &block_tables,
                                         const Tensor &sequence_lengths,
                                         float scale) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, query, kv_cache, block_tables,
                                          sequence_lengths);
    INFINICORE_GRAPH_OP_DISPATCH(output->device().getType(), output, query,
                                 kv_cache, block_tables, sequence_lengths, scale);
}

void FlashMlaDenseDecode::execute(Tensor output, const Tensor &query,
                                  const Tensor &kv_cache,
                                  const Tensor &block_tables,
                                  const Tensor &sequence_lengths,
                                  float scale) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(FlashMlaDenseDecode, output, query,
                                      kv_cache, block_tables, sequence_lengths,
                                      scale);
}

void flash_mla_sparse_prefill_(Tensor output, const Tensor &query,
                               const Tensor &kv_cache, const Tensor &indices,
                               const Tensor &topk_lens, float scale) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, query, kv_cache, indices,
                                          topk_lens);
    if (query->ndim() != 3 || output->ndim() != 3 || kv_cache->ndim() != 3 || indices->ndim() != 3 || topk_lens->ndim() != 1 || query->size(2) != 576 || output->size(2) != 512 || kv_cache->size(1) != 1 || kv_cache->size(2) != 576 || indices->size(1) != 1 || output->size(0) != query->size(0) || output->size(1) != query->size(1) || indices->size(0) != query->size(0) || topk_lens->numel() != query->size(0)) {
        throw std::runtime_error("flash_mla_sparse_prefill shape mismatch");
    }
    if (output->dtype() != DataType::BF16 || query->dtype() != DataType::BF16 || kv_cache->dtype() != DataType::BF16 || indices->dtype() != DataType::I32 || topk_lens->dtype() != DataType::I32) {
        throw std::runtime_error("flash_mla_sparse_prefill dtype mismatch");
    }
    FlashMlaSparsePrefill::execute(output, query, kv_cache, indices, topk_lens,
                                   scale);
}

void flash_mla_sparse_decode_(Tensor output, const Tensor &query,
                              const Tensor &kv_cache, const Tensor &indices,
                              const Tensor &topk_lens, float scale) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, query, kv_cache, indices,
                                          topk_lens);
    if (query->ndim() != 3 || output->ndim() != 3 || kv_cache->ndim() != 4 || indices->ndim() != 3 || topk_lens->ndim() != 1 || query->size(2) != 576 || output->size(2) != 512 || kv_cache->size(1) != 64 || kv_cache->size(2) != 1 || kv_cache->size(3) != 656 || indices->size(1) != 1 || output->size(0) != query->size(0) || output->size(1) != query->size(1) || indices->size(0) != query->size(0) || topk_lens->numel() != query->size(0)) {
        throw std::runtime_error("flash_mla_sparse_decode shape mismatch");
    }
    if (output->dtype() != DataType::BF16 || query->dtype() != DataType::BF16 || kv_cache->dtype() != DataType::U8 || indices->dtype() != DataType::I32 || topk_lens->dtype() != DataType::I32) {
        throw std::runtime_error("flash_mla_sparse_decode dtype mismatch");
    }
    FlashMlaSparseDecode::execute(output, query, kv_cache, indices, topk_lens,
                                  scale);
}

void flash_mla_dense_decode_(Tensor output, const Tensor &query,
                             const Tensor &kv_cache,
                             const Tensor &block_tables,
                             const Tensor &sequence_lengths, float scale) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, query, kv_cache, block_tables,
                                          sequence_lengths);
    if (query->ndim() != 4 || output->ndim() != 4 || kv_cache->ndim() != 4 || block_tables->ndim() != 2 || sequence_lengths->ndim() != 1 || query->size(1) != 1 || output->size(1) != 1 || query->size(3) != 576 || output->size(3) != 512 || kv_cache->size(1) != 64 || kv_cache->size(2) != 1 || kv_cache->size(3) != 576 || output->size(0) != query->size(0) || output->size(2) != query->size(2) || block_tables->size(0) != query->size(0) || sequence_lengths->numel() != query->size(0)) {
        throw std::runtime_error("flash_mla_dense_decode shape mismatch");
    }
    if (output->dtype() != query->dtype() || kv_cache->dtype() != query->dtype() || (query->dtype() != DataType::BF16 && query->dtype() != DataType::F16) || block_tables->dtype() != DataType::I32 || sequence_lengths->dtype() != DataType::I32) {
        throw std::runtime_error("flash_mla_dense_decode dtype mismatch");
    }
    if (query->stride(3) != 1 || output->stride(3) != 1 || kv_cache->stride(3) != 1 || block_tables->stride(1) != 1 || sequence_lengths->stride(0) != 1) {
        throw std::runtime_error(
            "flash_mla_dense_decode requires contiguous innermost dimensions");
    }
    FlashMlaDenseDecode::execute(output, query, kv_cache, block_tables,
                                 sequence_lengths, scale);
}

} // namespace infinicore::op
