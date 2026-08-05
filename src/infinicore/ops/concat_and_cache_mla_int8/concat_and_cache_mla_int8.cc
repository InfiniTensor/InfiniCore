#include "infinicore/ops/concat_and_cache_mla_int8.hpp"
#include "../../utils.hpp"

#include <stdexcept>

#include "../vendor_ops/vendor_ops_dispatch.hpp"

namespace infinicore::op {

namespace {

void validate_concat_and_cache_mla_int8(const Tensor &kv_c_int8,
                                        const Tensor &kv_c_scale,
                                        const Tensor &k_pe_int8,
                                        const Tensor &k_pe_scale,
                                        Tensor kv_cache,
                                        Tensor kv_cache_scale,
                                        const Tensor &slot_mapping) {
    if (!kv_c_int8 || !kv_c_scale || !k_pe_int8 || !k_pe_scale || !kv_cache || !kv_cache_scale || !slot_mapping) {
        throw std::runtime_error("concat_and_cache_mla_int8 expects non-empty tensors");
    }
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(kv_c_int8, kv_c_scale, k_pe_int8, k_pe_scale, kv_cache, kv_cache_scale, slot_mapping);
    if (kv_c_int8->dtype() != DataType::I8 || k_pe_int8->dtype() != DataType::I8 || kv_cache->dtype() != DataType::I8) {
        throw std::runtime_error("concat_and_cache_mla_int8 expects int8 kv_c_int8, k_pe_int8, and kv_cache");
    }
    if (kv_c_scale->dtype() != DataType::F32 || k_pe_scale->dtype() != DataType::F32 || kv_cache_scale->dtype() != DataType::F32) {
        throw std::runtime_error("concat_and_cache_mla_int8 expects float32 scales");
    }
    if (slot_mapping->dtype() != DataType::I64) {
        throw std::runtime_error("concat_and_cache_mla_int8 expects int64 flattened slot_mapping");
    }
    if (kv_c_int8->ndim() != 2 || k_pe_int8->ndim() != 2) {
        throw std::runtime_error("concat_and_cache_mla_int8 expects kv_c_int8/k_pe_int8 to be 2D [tokens, dim]");
    }
    const auto tokens = kv_c_int8->size(0);
    if (k_pe_int8->size(0) != tokens || slot_mapping->numel() != tokens) {
        throw std::runtime_error("concat_and_cache_mla_int8 expects matching token counts");
    }
    if (kv_c_scale->numel() != tokens || k_pe_scale->numel() != tokens) {
        throw std::runtime_error("concat_and_cache_mla_int8 expects one kv_c and k_pe scale per token");
    }
    const auto head_dim = kv_c_int8->size(1) + k_pe_int8->size(1);
    if (kv_cache->ndim() < 3 || kv_cache->size(kv_cache->ndim() - 1) != head_dim) {
        throw std::runtime_error("concat_and_cache_mla_int8 expects kv_cache last dim == kv_c_int8.shape[-1] + k_pe_int8.shape[-1]");
    }
    if (kv_cache_scale->ndim() < 3 || kv_cache_scale->size(kv_cache_scale->ndim() - 1) != 2) {
        throw std::runtime_error("concat_and_cache_mla_int8 expects kv_cache_scale last dim == 2");
    }
    if (!kv_c_int8->is_contiguous() || !kv_c_scale->is_contiguous() || !k_pe_int8->is_contiguous() || !k_pe_scale->is_contiguous()
        || !kv_cache->is_contiguous() || !kv_cache_scale->is_contiguous() || !slot_mapping->is_contiguous()) {
        throw std::runtime_error("concat_and_cache_mla_int8 expects contiguous tensors");
    }
}

} // namespace

void concat_and_cache_mla_int8_(const Tensor &kv_c_int8,
                                const Tensor &kv_c_scale,
                                const Tensor &k_pe_int8,
                                const Tensor &k_pe_scale,
                                Tensor kv_cache,
                                Tensor kv_cache_scale,
                                const Tensor &slot_mapping) {
    validate_concat_and_cache_mla_int8(kv_c_int8, kv_c_scale, k_pe_int8, k_pe_scale, kv_cache, kv_cache_scale, slot_mapping);

    auto kernel = vendor_ops::lookup(
        vendor_ops::concat_and_cache_mla_int8_dispatcher(),
        kv_cache->device().getType(),
        "concat_and_cache_mla_int8");
    kernel(kv_c_int8, kv_c_scale, k_pe_int8, k_pe_scale, kv_cache, kv_cache_scale, slot_mapping);
}

} // namespace infinicore::op
