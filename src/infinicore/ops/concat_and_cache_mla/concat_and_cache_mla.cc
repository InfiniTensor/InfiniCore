#include "infinicore/ops/concat_and_cache_mla.hpp"
#include "../../utils.hpp"

#include <stdexcept>

#if defined(ENABLE_ATEN) && defined(ENABLE_ILUVATAR_API)
#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/adaptor/vllm_iluvatar_adaptor.hpp"
#endif

namespace infinicore::op {

namespace {

void validate_concat_and_cache_mla(const Tensor &kv_c,
                                   const Tensor &k_pe,
                                   Tensor kv_cache,
                                   const Tensor &slot_mapping,
                                   const std::string &kv_cache_dtype,
                                   Tensor scale) {
    if (!kv_c || !k_pe || !kv_cache || !slot_mapping || !scale) {
        throw std::runtime_error("concat_and_cache_mla expects non-empty kv_c, k_pe, kv_cache, slot_mapping, and scale tensors");
    }
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(kv_c, k_pe, kv_cache, slot_mapping, scale);
    if (kv_cache_dtype != "auto" && kv_cache_dtype != "fp8"
        && kv_cache_dtype != "fp8_e4m3" && kv_cache_dtype != "fp8_e5m2"
        && kv_cache_dtype != "fp8_ds_mla") {
        throw std::runtime_error("concat_and_cache_mla expects kv_cache_dtype to be auto/fp8/fp8_e4m3/fp8_e5m2");
    }
    if (kv_c->ndim() != 2 || k_pe->ndim() != 2) {
        throw std::runtime_error("concat_and_cache_mla expects kv_c and k_pe to be 2D [tokens, dim]");
    }
    if (kv_c->size(0) != k_pe->size(0) || kv_c->size(0) != slot_mapping->numel()) {
        throw std::runtime_error("concat_and_cache_mla expects kv_c/k_pe tokens to match slot_mapping numel");
    }
    const auto head_dim = kv_c->size(1) + k_pe->size(1);
    if (kv_cache_dtype == "fp8_ds_mla") {
        const auto cache_stride = kv_c->size(1) + 4 * sizeof(float)
                                + k_pe->size(1) * sizeof(uint16_t);
        if (kv_cache->ndim() < 3 || kv_cache->dtype() != DataType::U8
            || kv_cache->size(kv_cache->ndim() - 1) != cache_stride) {
            throw std::runtime_error(
                "concat_and_cache_mla expects fp8_ds_mla uint8 cache with value/scales/rope layout");
        }
    } else if (kv_cache->ndim() < 3
               || kv_cache->size(kv_cache->ndim() - 1) != head_dim) {
        throw std::runtime_error(
            "concat_and_cache_mla expects kv_cache last dim == kv_c.shape[-1] + k_pe.shape[-1]");
    }
    if (slot_mapping->dtype() != DataType::I64 && slot_mapping->dtype() != DataType::I32) {
        throw std::runtime_error("concat_and_cache_mla expects slot_mapping dtype int64 or int32");
    }
    if (scale->dtype() != DataType::F32) {
        throw std::runtime_error("concat_and_cache_mla expects scale dtype float32");
    }
    if (!kv_c->is_contiguous() || !k_pe->is_contiguous() || !kv_cache->is_contiguous() || !slot_mapping->is_contiguous() || !scale->is_contiguous()) {
        throw std::runtime_error("concat_and_cache_mla expects contiguous tensors");
    }
}

} // namespace

void concat_and_cache_mla_(const Tensor &kv_c,
                           const Tensor &k_pe,
                           Tensor kv_cache,
                           const Tensor &slot_mapping,
                           const std::string &kv_cache_dtype,
                           Tensor scale) {
    validate_concat_and_cache_mla(kv_c, k_pe, kv_cache, slot_mapping, kv_cache_dtype, scale);

#if defined(ENABLE_ATEN) && defined(ENABLE_ILUVATAR_API)
    if (kv_cache->device().getType() == Device::Type::ILUVATAR) {
        if (!adaptor::vllm_iluvatar::concat_and_cache_mla_available()) {
            throw std::runtime_error("concat_and_cache_mla requires vllm_iluvatar perf extension on Iluvatar");
        }
        auto kv_c_at = adaptor::to_aten_tensor(kv_c);
        auto k_pe_at = adaptor::to_aten_tensor(k_pe);
        auto kv_cache_at = adaptor::to_aten_tensor(kv_cache);
        auto slot_mapping_at = adaptor::to_aten_tensor(slot_mapping);
        auto scale_at = adaptor::to_aten_tensor(scale);
        adaptor::vllm_iluvatar::concat_and_cache_mla(kv_c_at, k_pe_at, kv_cache_at, slot_mapping_at, kv_cache_dtype, scale_at);
        return;
    }
#endif

    throw std::runtime_error("concat_and_cache_mla currently supports only Iluvatar builds with ATen and vllm_iluvatar");
}

} // namespace infinicore::op
