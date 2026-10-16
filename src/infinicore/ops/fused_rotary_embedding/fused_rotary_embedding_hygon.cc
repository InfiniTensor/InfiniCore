#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/adaptor/hygon_aiter_adaptor.hpp"

#include "../vendor_ops/vendor_ops_dispatch.hpp"

#include <stdexcept>

namespace infinicore::op::fused_rotary_embedding_impl::hygon {

void run(Tensor query,
         Tensor key,
         const Tensor &positions,
         int64_t head_size,
         const Tensor &cos_sin_cache,
         bool is_neox) {
    if (!adaptor::hygon_aiter::rotary_embedding_available()) {
        throw std::runtime_error(
            "fused_rotary_embedding requires AITER on Hygon");
    }
    const size_t cache_dim = cos_sin_cache->size(1) / 2;
    // AITER's kernel assumes each cache row has exactly cache_dim elements.
    // Narrow keeps the packed [cos | sin] tensor's wider row stride, so make
    // both halves dense before crossing the vendor boundary.
    auto cos_cache = cos_sin_cache->narrow({{1, 0, cache_dim}})->contiguous();
    auto sin_cache = cos_sin_cache->narrow({{1, cache_dim, cache_dim}})->contiguous();
    auto positions_at = adaptor::to_aten_tensor(positions);
    auto query_at = adaptor::to_aten_tensor(query);
    auto key_at = adaptor::to_aten_tensor(key);
    auto cos_cache_at = adaptor::to_aten_tensor(cos_cache);
    auto sin_cache_at = adaptor::to_aten_tensor(sin_cache);
    adaptor::hygon_aiter::rotary_embedding(
        positions_at,
        query_at,
        key_at,
        head_size,
        cos_cache_at,
        sin_cache_at,
        is_neox,
        false);
}

static bool registered = []() {
    vendor_ops::fused_rotary_embedding_dispatcher().registerDevice(
        Device::Type::HYGON, &run);
    return true;
}();

} // namespace infinicore::op::fused_rotary_embedding_impl::hygon
