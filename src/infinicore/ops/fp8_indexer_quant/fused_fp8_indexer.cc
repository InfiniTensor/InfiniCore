#include "infinicore/ops/fp8_indexer_quant.hpp"

#include "../../utils.hpp"
#include <stdexcept>

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(FusedFp8Indexer);

FusedFp8Indexer::FusedFp8Indexer(
    Tensor q_fp8, Tensor weights_fp32, Tensor k_cache,
    const Tensor &q_raw, const Tensor &k_weights,
    const Tensor &norm_weight, const Tensor &norm_bias,
    const Tensor &positions, const Tensor &cos_sin_cache,
    const Tensor &slot_mapping, size_t rope_dim,
    double eps, double weights_scale) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(
        q_fp8, weights_fp32, k_cache, q_raw, k_weights, norm_weight,
        norm_bias, positions, cos_sin_cache, slot_mapping);
    INFINICORE_GRAPH_OP_DISPATCH(
        q_raw->device().getType(), q_fp8, weights_fp32, k_cache,
        q_raw, k_weights, norm_weight, norm_bias, positions,
        cos_sin_cache, slot_mapping, rope_dim, eps, weights_scale);
}

void FusedFp8Indexer::execute(
    Tensor q_fp8, Tensor weights_fp32, Tensor k_cache,
    const Tensor &q_raw, const Tensor &k_weights,
    const Tensor &norm_weight, const Tensor &norm_bias,
    const Tensor &positions, const Tensor &cos_sin_cache,
    const Tensor &slot_mapping, size_t rope_dim,
    double eps, double weights_scale) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(
        FusedFp8Indexer, q_fp8, weights_fp32, k_cache,
        q_raw, k_weights, norm_weight, norm_bias, positions,
        cos_sin_cache, slot_mapping, rope_dim, eps, weights_scale);
}

void fused_fp8_indexer_(
    Tensor q_fp8, Tensor weights_fp32, Tensor k_cache,
    const Tensor &q_raw, const Tensor &k_weights,
    const Tensor &norm_weight, const Tensor &norm_bias,
    const Tensor &positions, const Tensor &cos_sin_cache,
    const Tensor &slot_mapping, size_t rope_dim,
    double eps, double weights_scale) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(
        q_fp8, weights_fp32, k_cache, q_raw, k_weights, norm_weight,
        norm_bias, positions, cos_sin_cache, slot_mapping);
    if (q_raw->ndim() != 3 || k_weights->ndim() != 2
        || weights_fp32->ndim() != 2 || k_cache->ndim() != 3
        || norm_weight->ndim() != 1 || norm_bias->ndim() != 1
        || positions->ndim() != 1 || cos_sin_cache->ndim() != 2
        || slot_mapping->ndim() != 1
        || q_fp8->shape() != q_raw->shape()
        || weights_fp32->size(0) != q_raw->size(0)
        || weights_fp32->size(1) != q_raw->size(1)
        || k_weights->size(0) != q_raw->size(0)
        || k_weights->size(1) != q_raw->size(2) + q_raw->size(1)
        || norm_weight->numel() != q_raw->size(2)
        || norm_bias->numel() != q_raw->size(2)
        || positions->numel() != q_raw->size(0)
        || slot_mapping->numel() != q_raw->size(0)
        || cos_sin_cache->size(1) != rope_dim
        || k_cache->size(2) != q_raw->size(2) + sizeof(float)) {
        throw std::runtime_error("fused_fp8_indexer shape mismatch");
    }
    if ((q_raw->dtype() != DataType::F16
         && q_raw->dtype() != DataType::BF16)
        || k_weights->dtype() != q_raw->dtype()
        || norm_weight->dtype() != q_raw->dtype()
        || norm_bias->dtype() != q_raw->dtype()
        || cos_sin_cache->dtype() != q_raw->dtype()
        || q_fp8->dtype() != DataType::F8
        || weights_fp32->dtype() != DataType::F32
        || k_cache->dtype() != DataType::U8
        || positions->dtype() != DataType::I64
        || slot_mapping->dtype() != DataType::I64) {
        throw std::runtime_error("fused_fp8_indexer dtype mismatch");
    }
    for (const auto &tensor : {
             q_fp8, weights_fp32, k_cache, q_raw, k_weights,
             norm_weight, norm_bias, positions, cos_sin_cache, slot_mapping}) {
        if (!tensor->is_contiguous()) {
            throw std::runtime_error(
                "fused_fp8_indexer expects contiguous tensors");
        }
    }
    if (q_raw->size(2) != 128 || rope_dim == 0
        || rope_dim > q_raw->size(2) || rope_dim % 2 != 0
        || eps <= 0.0 || weights_scale <= 0.0) {
        throw std::runtime_error("fused_fp8_indexer invalid parameters");
    }
    FusedFp8Indexer::execute(
        q_fp8, weights_fp32, k_cache, q_raw, k_weights,
        norm_weight, norm_bias, positions, cos_sin_cache,
        slot_mapping, rope_dim, eps, weights_scale);
}

} // namespace infinicore::op
