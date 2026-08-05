#if defined(ENABLE_ILUVATAR_VENDOR_OPS)
#include "../../vendor_ops/vendor_ops_dispatch.hpp"

#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/adaptor/iluvatar_vendor_adaptor.hpp"

#include <stdexcept>

namespace infinicore::op::concat_and_cache_mla_impl::iluvatar {

void run(const Tensor &kv_c,
         const Tensor &k_pe,
         Tensor kv_cache,
         const Tensor &slot_mapping,
         const std::string &kv_cache_dtype,
         Tensor scale) {
    if (!adaptor::iluvatar_vendor::concat_and_cache_mla_available()) {
        throw std::runtime_error("concat_and_cache_mla requires the Iluvatar vendor extension");
    }
    auto kv_c_at = adaptor::to_aten_tensor(kv_c);
    auto k_pe_at = adaptor::to_aten_tensor(k_pe);
    auto kv_cache_at = adaptor::to_aten_tensor(kv_cache);
    auto slot_mapping_at = adaptor::to_aten_tensor(slot_mapping);
    auto scale_at = adaptor::to_aten_tensor(scale);
    adaptor::iluvatar_vendor::concat_and_cache_mla(
        kv_c_at, k_pe_at,
        kv_cache_at, slot_mapping_at,
        kv_cache_dtype, scale_at);
}

static bool registered = []() {
    vendor_ops::concat_and_cache_mla_dispatcher().registerDevice(Device::Type::ILUVATAR, &run);
    return true;
}();

} // namespace infinicore::op::concat_and_cache_mla_impl::iluvatar
#endif
