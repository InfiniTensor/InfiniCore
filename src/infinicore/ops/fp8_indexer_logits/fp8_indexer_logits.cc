#include "infinicore/ops/fp8_indexer_logits.hpp"

#include "../../utils.hpp"
#include <stdexcept>

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Fp8IndexerLogits);

Fp8IndexerLogits::Fp8IndexerLogits(
    Tensor logits,
    const Tensor &q_fp8,
    const Tensor &kv_cache,
    const Tensor &block_tables,
    const Tensor &weights_fp32,
    const Tensor &positions,
    const Tensor &request_ids) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(
        logits, q_fp8, kv_cache, block_tables, weights_fp32, positions, request_ids);
    INFINICORE_GRAPH_OP_DISPATCH(
        q_fp8->device().getType(), logits, q_fp8, kv_cache, block_tables,
        weights_fp32, positions, request_ids);
}

void Fp8IndexerLogits::execute(
    Tensor logits,
    const Tensor &q_fp8,
    const Tensor &kv_cache,
    const Tensor &block_tables,
    const Tensor &weights_fp32,
    const Tensor &positions,
    const Tensor &request_ids) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(
        Fp8IndexerLogits, logits, q_fp8, kv_cache, block_tables,
        weights_fp32, positions, request_ids);
}

void fp8_indexer_logits_(
    Tensor logits,
    const Tensor &q_fp8,
    const Tensor &kv_cache,
    const Tensor &block_tables,
    const Tensor &weights_fp32,
    const Tensor &positions,
    const Tensor &request_ids) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(
        logits, q_fp8, kv_cache, block_tables, weights_fp32, positions, request_ids);
    if (logits->ndim() != 2 || q_fp8->ndim() != 3 || kv_cache->ndim() != 3
        || block_tables->ndim() != 2 || weights_fp32->ndim() != 2
        || positions->ndim() != 1 || request_ids->ndim() != 1
        || logits->size(0) != q_fp8->size(0)
        || weights_fp32->size(0) != q_fp8->size(0)
        || weights_fp32->size(1) != q_fp8->size(1)
        || positions->numel() != q_fp8->size(0)
        || request_ids->numel() != q_fp8->size(0)
        || kv_cache->size(1) != 64 || q_fp8->size(2) != 128
        || kv_cache->size(2) != q_fp8->size(2) + sizeof(float)) {
        throw std::runtime_error("fp8_indexer_logits shape mismatch");
    }
    if (logits->dtype() != DataType::F32 || q_fp8->dtype() != DataType::F8
        || kv_cache->dtype() != DataType::U8
        || block_tables->dtype() != DataType::I32
        || weights_fp32->dtype() != DataType::F32
        || positions->dtype() != DataType::I64
        || request_ids->dtype() != DataType::I32) {
        throw std::runtime_error("fp8_indexer_logits dtype mismatch");
    }
    for (const auto &tensor : {logits, q_fp8, kv_cache, block_tables,
                               weights_fp32, positions, request_ids}) {
        if (!tensor->is_contiguous()) {
            throw std::runtime_error("fp8_indexer_logits expects contiguous tensors");
        }
    }
    Fp8IndexerLogits::execute(
        logits, q_fp8, kv_cache, block_tables, weights_fp32, positions, request_ids);
}

} // namespace infinicore::op
