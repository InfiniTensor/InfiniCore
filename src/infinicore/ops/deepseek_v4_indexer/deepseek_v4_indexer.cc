#include "infinicore/ops/deepseek_v4_indexer.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(DeepseekV4Indexer);

DeepseekV4Indexer::DeepseekV4Indexer(Tensor indices,
                                     const Tensor &q,
                                     const Tensor &weights,
                                     const Tensor &compressed,
                                     const Tensor &positions,
                                     size_t query_start,
                                     size_t compress_ratio) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(indices, q, weights, compressed, positions);
    INFINICORE_GRAPH_OP_DISPATCH(
        indices->device().getType(),
        indices,
        q,
        weights,
        compressed,
        positions,
        query_start,
        compress_ratio);
}

void DeepseekV4Indexer::execute(Tensor indices,
                                const Tensor &q,
                                const Tensor &weights,
                                const Tensor &compressed,
                                const Tensor &positions,
                                size_t query_start,
                                size_t compress_ratio) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(
        DeepseekV4Indexer,
        indices,
        q,
        weights,
        compressed,
        positions,
        query_start,
        compress_ratio);
}

Tensor deepseek_v4_indexer(const Tensor &q,
                           const Tensor &weights,
                           const Tensor &compressed,
                           const Tensor &positions,
                           size_t topk,
                           size_t query_start,
                           size_t compress_ratio) {
    const auto &q_shape = q->shape();
    INFINICORE_ASSERT(q_shape.size() == 4);
    auto indices = Tensor::empty({q_shape[0], q_shape[1], topk}, DataType::I64, q->device());
    deepseek_v4_indexer_(indices, q, weights, compressed, positions, query_start, compress_ratio);
    return indices;
}

void deepseek_v4_indexer_(Tensor indices,
                          const Tensor &q,
                          const Tensor &weights,
                          const Tensor &compressed,
                          const Tensor &positions,
                          size_t query_start,
                          size_t compress_ratio) {
    DeepseekV4Indexer::execute(indices, q, weights, compressed, positions, query_start, compress_ratio);
}

} // namespace infinicore::op
