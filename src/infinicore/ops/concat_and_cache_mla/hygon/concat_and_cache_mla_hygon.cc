#if defined(ENABLE_HYGON_API)
#include "../../infiniop_impl.hpp"
#include "../../vendor_ops/vendor_ops_dispatch.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/ops/concat_and_cache_mla.hpp"
#include "infiniop/ops/concat_and_cache_mla.h"

#include <stdexcept>

namespace infinicore::op::concat_and_cache_mla_impl::hygon {

void run(const Tensor &kv_c,
         const Tensor &k_pe,
         Tensor kv_cache,
         const Tensor &slot_mapping,
         const std::string &kv_cache_dtype,
         Tensor scale) {
    (void)scale;
    if (kv_cache_dtype != "auto") {
        throw std::runtime_error(
            "Hygon concat_and_cache_mla currently supports BF16/FP16 auto cache only");
    }
    INFINICORE_CHECK_ERROR(infiniopConcatAndCacheMla(
        context::getInfiniopHandle(kv_cache->device()),
        kv_c->desc(),
        k_pe->desc(),
        kv_cache->desc(),
        slot_mapping->desc(),
        kv_c->data(),
        k_pe->data(),
        kv_cache->data(),
        slot_mapping->data(),
        context::getStream()));
}

static bool registered = []() {
    vendor_ops::concat_and_cache_mla_dispatcher().registerDevice(Device::Type::HYGON, &run);
    return true;
}();

} // namespace infinicore::op::concat_and_cache_mla_impl::hygon
#endif
