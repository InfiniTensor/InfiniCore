#if defined(ENABLE_HYGON_API) && defined(ENABLE_VENDOR_OPS)
#include "../../infiniop_impl.hpp"
#include "../../vendor_ops/vendor_ops_dispatch.hpp"
#include "infinicore/context/context.hpp"
#include "infiniop/ops/scaled_mm_w4a8.h"

namespace infinicore::op::prepare_glm_w4a16_awq_impl::hygon {
void run(Tensor qweight, Tensor qzeros, Tensor scales,
         const Tensor &checkpoint_weight, const Tensor &channel_scales) {
    INFINICORE_CHECK_ERROR(infiniopPrepareGlmW4A16Awq(
        context::getInfiniopHandle(qweight->device()), qweight->desc(),
        qzeros->desc(), scales->desc(), checkpoint_weight->desc(),
        channel_scales->desc(), qweight->data(), qzeros->data(), scales->data(),
        checkpoint_weight->data(), channel_scales->data(), context::getStream()));
}

static bool registered = []() {
    vendor_ops::prepare_glm_w4a16_awq_dispatcher().registerDevice(
        Device::Type::HYGON, &run);
    return true;
}();
} // namespace infinicore::op::prepare_glm_w4a16_awq_impl::hygon
#endif
