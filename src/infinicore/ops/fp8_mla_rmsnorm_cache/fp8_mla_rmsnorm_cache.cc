#include "infinicore/ops/fp8_mla_rmsnorm_cache.hpp"

#include "../../utils.hpp"
#include <stdexcept>

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Fp8MlaRmsnormCache);
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Fp8MlaRmsnormDualCache);

Fp8MlaRmsnormCache::Fp8MlaRmsnormCache(
    Tensor cache,
    const Tensor &compressed_kv,
    const Tensor &norm_weight,
    const Tensor &rope,
    const Tensor &slot_mapping,
    double eps) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(
        cache, compressed_kv, norm_weight, rope, slot_mapping);
    INFINICORE_GRAPH_OP_DISPATCH(
        cache->device().getType(), cache, compressed_kv, norm_weight,
        rope, slot_mapping, eps);
}

void Fp8MlaRmsnormCache::execute(
    Tensor cache,
    const Tensor &compressed_kv,
    const Tensor &norm_weight,
    const Tensor &rope,
    const Tensor &slot_mapping,
    double eps) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(
        Fp8MlaRmsnormCache, cache, compressed_kv, norm_weight,
        rope, slot_mapping, eps);
}

Fp8MlaRmsnormDualCache::Fp8MlaRmsnormDualCache(
    Tensor cache,
    Tensor vendor_cache,
    const Tensor &compressed_kv,
    const Tensor &norm_weight,
    const Tensor &rope,
    const Tensor &slot_mapping,
    double eps) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(
        cache, vendor_cache, compressed_kv, norm_weight, rope, slot_mapping);
    INFINICORE_GRAPH_OP_DISPATCH(
        cache->device().getType(), cache, vendor_cache, compressed_kv,
        norm_weight, rope, slot_mapping, eps);
}

void Fp8MlaRmsnormDualCache::execute(
    Tensor cache,
    Tensor vendor_cache,
    const Tensor &compressed_kv,
    const Tensor &norm_weight,
    const Tensor &rope,
    const Tensor &slot_mapping,
    double eps) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(
        Fp8MlaRmsnormDualCache, cache, vendor_cache, compressed_kv,
        norm_weight, rope, slot_mapping, eps);
}

void fp8_mla_rmsnorm_cache_(
    Tensor cache,
    const Tensor &compressed_kv,
    const Tensor &norm_weight,
    const Tensor &rope,
    const Tensor &slot_mapping,
    double eps) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(
        cache, compressed_kv, norm_weight, rope, slot_mapping);
    if (cache->ndim() != 3 || compressed_kv->ndim() != 2
        || norm_weight->ndim() != 1 || rope->ndim() != 2
        || slot_mapping->ndim() != 1
        || compressed_kv->size(1) != 512
        || norm_weight->numel() != 512
        || rope->size(0) != compressed_kv->size(0)
        || rope->size(1) != 64
        || slot_mapping->numel() != compressed_kv->size(0)
        || cache->size(2) != 656) {
        throw std::runtime_error("fp8_mla_rmsnorm_cache shape mismatch");
    }
    if (cache->dtype() != DataType::U8
        || compressed_kv->dtype() != DataType::BF16
        || norm_weight->dtype() != DataType::BF16
        || rope->dtype() != DataType::BF16
        || slot_mapping->dtype() != DataType::I64) {
        throw std::runtime_error("fp8_mla_rmsnorm_cache dtype mismatch");
    }
    for (const auto &tensor : {
             cache, compressed_kv, norm_weight, rope, slot_mapping}) {
        if (!tensor->is_contiguous()) {
            throw std::runtime_error(
                "fp8_mla_rmsnorm_cache expects contiguous tensors");
        }
    }
    if (eps <= 0.0) {
        throw std::runtime_error("fp8_mla_rmsnorm_cache requires eps > 0");
    }
    Fp8MlaRmsnormCache::execute(
        cache, compressed_kv, norm_weight, rope, slot_mapping, eps);
}

void fp8_mla_rmsnorm_dual_cache_(
    Tensor cache,
    Tensor vendor_cache,
    const Tensor &compressed_kv,
    const Tensor &norm_weight,
    const Tensor &rope,
    const Tensor &slot_mapping,
    double eps) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(
        cache, vendor_cache, compressed_kv, norm_weight, rope, slot_mapping);
    if (cache->ndim() != 3 || vendor_cache->ndim() != 3
        || compressed_kv->ndim() != 2 || norm_weight->ndim() != 1
        || rope->ndim() != 2 || slot_mapping->ndim() != 1
        || compressed_kv->size(1) != 512
        || norm_weight->numel() != 512
        || rope->size(0) != compressed_kv->size(0)
        || rope->size(1) != 64
        || slot_mapping->numel() != compressed_kv->size(0)
        || cache->size(2) != 656
        || vendor_cache->size(0) != cache->size(0)
        || vendor_cache->size(1) != cache->size(1)
        || vendor_cache->size(2) != 576) {
        throw std::runtime_error("fp8_mla_rmsnorm_dual_cache shape mismatch");
    }
    if (cache->dtype() != DataType::U8
        || vendor_cache->dtype() != DataType::BF16
        || compressed_kv->dtype() != DataType::BF16
        || norm_weight->dtype() != DataType::BF16
        || rope->dtype() != DataType::BF16
        || slot_mapping->dtype() != DataType::I64) {
        throw std::runtime_error("fp8_mla_rmsnorm_dual_cache dtype mismatch");
    }
    for (const auto &tensor : {
             cache, vendor_cache, compressed_kv, norm_weight, rope,
             slot_mapping}) {
        if (!tensor->is_contiguous()) {
            throw std::runtime_error(
                "fp8_mla_rmsnorm_dual_cache expects contiguous tensors");
        }
    }
    if (eps <= 0.0) {
        throw std::runtime_error(
            "fp8_mla_rmsnorm_dual_cache requires eps > 0");
    }
    Fp8MlaRmsnormDualCache::execute(
        cache, vendor_cache, compressed_kv, norm_weight, rope,
        slot_mapping, eps);
}

} // namespace infinicore::op
