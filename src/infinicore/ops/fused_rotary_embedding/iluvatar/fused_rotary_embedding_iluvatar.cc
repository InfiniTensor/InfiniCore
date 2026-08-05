#if defined(ENABLE_ILUVATAR_VENDOR_OPS)
#include "../../vendor_ops/vendor_ops_dispatch.hpp"

#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/adaptor/iluvatar_vendor_adaptor.hpp"

#include <stdexcept>

namespace infinicore::op::fused_rotary_embedding_impl::iluvatar {

void run(Tensor query,
         Tensor key,
         const Tensor &positions,
         int64_t head_size,
         const Tensor &cos_sin_cache,
         bool is_neox) {
    if (!adaptor::iluvatar_vendor::rotary_embedding_available()) {
        throw std::runtime_error("fused_rotary_embedding requires the Iluvatar vendor extension");
    }
    auto positions_at = adaptor::to_aten_tensor(positions);
    auto query_at = adaptor::to_aten_tensor(query);
    auto key_at = adaptor::to_aten_tensor(key);
    auto cos_sin_cache_at = adaptor::to_aten_tensor(cos_sin_cache);
    adaptor::iluvatar_vendor::rotary_embedding(
        positions_at, query_at,
        std::optional<at::Tensor>(key_at), head_size,
        cos_sin_cache_at, is_neox);
}

static bool registered = []() {
    vendor_ops::fused_rotary_embedding_dispatcher().registerDevice(Device::Type::ILUVATAR, &run);
    return true;
}();

} // namespace infinicore::op::fused_rotary_embedding_impl::iluvatar
#endif
