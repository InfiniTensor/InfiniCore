#if defined(ENABLE_ATEN) && defined(ENABLE_HYGON_API)

#include "../../vendor_ops/vendor_ops_dispatch.hpp"

#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/adaptor/hygon_lightop_adaptor.hpp"

#include <stdexcept>

namespace infinicore::op::concat_and_cache_mla_impl::hygon {

void run(const Tensor &kv_c,
         const Tensor &k_pe,
         Tensor kv_cache,
         const Tensor &slot_mapping,
         const std::string &kv_cache_dtype,
         Tensor scale) {
    (void)scale;
    if (kv_cache_dtype == "auto") {
        auto kv_c_at = adaptor::to_aten_tensor(kv_c);
        auto k_pe_at = adaptor::to_aten_tensor(k_pe);
        auto kv_cache_at = adaptor::to_aten_tensor(kv_cache);
        auto slot_mapping_at = adaptor::to_aten_tensor(slot_mapping);
        if (slot_mapping_at.scalar_type() != at::ScalarType::Long) {
            slot_mapping_at = slot_mapping_at.to(at::ScalarType::Long);
        }

        const auto kv_dim = kv_c_at.size(1);
        const auto rope_dim = k_pe_at.size(1);
        auto cache_rows = kv_cache_at.view({-1, kv_dim + rope_dim});
        cache_rows.narrow(1, 0, kv_dim).index_copy_(0, slot_mapping_at, kv_c_at);
        cache_rows.narrow(1, kv_dim, rope_dim).index_copy_(0, slot_mapping_at, k_pe_at);
        return;
    }

    if (!adaptor::hygon_lightop::fused_quantize_and_store_mla_kv_cache_available()) {
        throw std::runtime_error(
            "concat_and_cache_mla requires LightOp on Hygon");
    }
    auto kv_c_at = adaptor::to_aten_tensor(kv_c);
    auto k_pe_at = adaptor::to_aten_tensor(k_pe);
    auto kv_cache_at = adaptor::to_aten_tensor(kv_cache);
    auto slot_mapping_at = adaptor::to_aten_tensor(slot_mapping);
    adaptor::hygon_lightop::fused_quantize_and_store_mla_kv_cache(
        kv_c_at,
        k_pe_at,
        kv_cache_at,
        slot_mapping_at,
        kv_cache_dtype == "fp8_ds_mla" ? "fp8_e4m3" : kv_cache_dtype,
        1e-6f);
}

static bool registered = []() {
    vendor_ops::concat_and_cache_mla_dispatcher().registerDevice(
        Device::Type::HYGON, &run);
    return true;
}();

} // namespace infinicore::op::concat_and_cache_mla_impl::hygon

#endif
