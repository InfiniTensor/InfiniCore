#if defined(ENABLE_ILUVATAR_VENDOR_OPS)
#include "../../vendor_ops/vendor_ops_dispatch.hpp"

#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/adaptor/iluvatar_vendor_adaptor.hpp"

#include <stdexcept>

namespace infinicore::op::concat_and_cache_mla_int8_impl::iluvatar {

void run(const Tensor &kv_c_int8,
         const Tensor &kv_c_scale,
         const Tensor &k_pe_int8,
         const Tensor &k_pe_scale,
         Tensor kv_cache,
         Tensor kv_cache_scale,
         const Tensor &slot_mapping) {
    if (!adaptor::iluvatar_vendor::concat_and_cache_mla_int8_available()) {
        throw std::runtime_error("concat_and_cache_mla_int8 requires the Iluvatar vendor extension");
    }
    auto kv_c_int8_at = adaptor::to_aten_tensor(kv_c_int8);
    auto kv_c_scale_at = adaptor::to_aten_tensor(kv_c_scale);
    auto k_pe_int8_at = adaptor::to_aten_tensor(k_pe_int8);
    auto k_pe_scale_at = adaptor::to_aten_tensor(k_pe_scale);
    auto kv_cache_at = adaptor::to_aten_tensor(kv_cache);
    auto kv_cache_scale_at = adaptor::to_aten_tensor(kv_cache_scale);
    auto slot_mapping_at = adaptor::to_aten_tensor(slot_mapping);
    adaptor::iluvatar_vendor::concat_and_cache_mla_int8(
        kv_c_int8_at, kv_c_scale_at,
        k_pe_int8_at, k_pe_scale_at,
        kv_cache_at, kv_cache_scale_at,
        slot_mapping_at);
}

static bool registered = []() {
    vendor_ops::concat_and_cache_mla_int8_dispatcher().registerDevice(Device::Type::ILUVATAR, &run);
    return true;
}();

} // namespace infinicore::op::concat_and_cache_mla_int8_impl::iluvatar
#endif
