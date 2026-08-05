#include "infinicore/ops/fp8_indexer_quant.hpp"

#include "../../utils.hpp"
#include <stdexcept>

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Fp8IndexerQuant);

Fp8IndexerQuant::Fp8IndexerQuant(
    Tensor q_fp8,
    Tensor weights_fp32,
    const Tensor &q,
    const Tensor &weights) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(q_fp8, weights_fp32, q, weights);
    INFINICORE_GRAPH_OP_DISPATCH(
        q->device().getType(), q_fp8, weights_fp32, q, weights);
}

void Fp8IndexerQuant::execute(
    Tensor q_fp8,
    Tensor weights_fp32,
    const Tensor &q,
    const Tensor &weights) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(
        Fp8IndexerQuant, q_fp8, weights_fp32, q, weights);
}

void fp8_indexer_quant_(
    Tensor q_fp8,
    Tensor weights_fp32,
    const Tensor &q,
    const Tensor &weights) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(q_fp8, weights_fp32, q, weights);
    if (q->ndim() != 3 || weights->ndim() != 2
        || q_fp8->shape() != q->shape()
        || weights_fp32->shape() != weights->shape()
        || q->size(0) != weights->size(0)
        || q->size(1) != weights->size(1)) {
        throw std::runtime_error("fp8_indexer_quant shape mismatch");
    }
    if ((q->dtype() != DataType::F16 && q->dtype() != DataType::BF16)
        || weights->dtype() != q->dtype()
        || q_fp8->dtype() != DataType::F8
        || weights_fp32->dtype() != DataType::F32) {
        throw std::runtime_error("fp8_indexer_quant dtype mismatch");
    }
    if (!q->is_contiguous() || !weights->is_contiguous()
        || !q_fp8->is_contiguous() || !weights_fp32->is_contiguous()) {
        throw std::runtime_error("fp8_indexer_quant expects contiguous tensors");
    }
    Fp8IndexerQuant::execute(q_fp8, weights_fp32, q, weights);
}

} // namespace infinicore::op
