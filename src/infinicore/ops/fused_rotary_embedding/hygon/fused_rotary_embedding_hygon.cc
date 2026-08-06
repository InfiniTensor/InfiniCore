#if defined(ENABLE_HYGON_API)
#include "../../infiniop_impl.hpp"
#include "../../vendor_ops/vendor_ops_dispatch.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/ops/fused_rotary_embedding.hpp"
#include "infiniop/ops/fused_rotary_embedding.h"

namespace infinicore::op::fused_rotary_embedding_impl::hygon {

void run(Tensor query,
         Tensor key,
         const Tensor &positions,
         int64_t head_size,
         const Tensor &cos_sin_cache,
         bool is_neox) {
    INFINICORE_CHECK_ERROR(infiniopFusedRotaryEmbedding(
        context::getInfiniopHandle(query->device()),
        query->desc(),
        key->desc(),
        positions->desc(),
        cos_sin_cache->desc(),
        query->data(),
        key->data(),
        positions->data(),
        cos_sin_cache->data(),
        head_size,
        is_neox,
        context::getStream()));
}

static bool registered = []() {
    vendor_ops::fused_rotary_embedding_dispatcher().registerDevice(Device::Type::HYGON, &run);
    return true;
}();

} // namespace infinicore::op::fused_rotary_embedding_impl::hygon
#endif
