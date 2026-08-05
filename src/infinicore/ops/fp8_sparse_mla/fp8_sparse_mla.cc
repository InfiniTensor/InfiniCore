#include "infinicore/ops/fp8_sparse_mla.hpp"

#include "../../utils.hpp"
#include <stdexcept>

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Fp8SparseMla);

Fp8SparseMla::Fp8SparseMla(
    Tensor output,
    const Tensor &query,
    const Tensor &kv_cache,
    const Tensor &indices,
    const Tensor &topk_lens,
    float scale) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(
        output, query, kv_cache, indices, topk_lens);
    INFINICORE_GRAPH_OP_DISPATCH(
        query->device().getType(), output, query, kv_cache, indices, topk_lens, scale);
}

void Fp8SparseMla::execute(
    Tensor output,
    const Tensor &query,
    const Tensor &kv_cache,
    const Tensor &indices,
    const Tensor &topk_lens,
    float scale) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(
        Fp8SparseMla, output, query, kv_cache, indices, topk_lens, scale);
}

void fp8_sparse_mla_(
    Tensor output,
    const Tensor &query,
    const Tensor &kv_cache,
    const Tensor &indices,
    const Tensor &topk_lens,
    float scale) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(
        output, query, kv_cache, indices, topk_lens);
    if (output->ndim() != 3 || query->ndim() != 3 || kv_cache->ndim() != 3
        || indices->ndim() != 3 || topk_lens->ndim() != 1
        || output->size(0) != query->size(0)
        || output->size(1) != query->size(1)
        || output->size(2) != 512 || query->size(2) != 576
        || kv_cache->size(1) != 1 || kv_cache->size(2) != 656
        || indices->size(0) != query->size(0) || indices->size(1) != 1
        || topk_lens->numel() != query->size(0)) {
        throw std::runtime_error("fp8_sparse_mla shape mismatch");
    }
    if (output->dtype() != DataType::BF16 || query->dtype() != DataType::BF16
        || kv_cache->dtype() != DataType::U8
        || indices->dtype() != DataType::I32
        || topk_lens->dtype() != DataType::I32) {
        throw std::runtime_error("fp8_sparse_mla dtype mismatch");
    }
    for (const auto &tensor : {output, query, kv_cache, indices, topk_lens}) {
        if (!tensor->is_contiguous()) {
            throw std::runtime_error("fp8_sparse_mla expects contiguous tensors");
        }
    }
    Fp8SparseMla::execute(output, query, kv_cache, indices, topk_lens, scale);
}

} // namespace infinicore::op
