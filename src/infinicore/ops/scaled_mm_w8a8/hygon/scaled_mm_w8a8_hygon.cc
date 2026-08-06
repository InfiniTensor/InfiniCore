#if defined(ENABLE_HYGON_API)
#include "../../infiniop_impl.hpp"
#include "../../vendor_ops/vendor_ops_dispatch.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/ops/scaled_mm_w8a8.hpp"
#include "infiniop/ops/scaled_mm_w8a8.h"

namespace infinicore::op::scaled_mm_w8a8_impl::hygon {

void run(Tensor out,
         const Tensor &a,
         const Tensor &b,
         const Tensor &a_scales,
         const Tensor &b_scales,
         std::optional<Tensor> bias,
         bool trans_weight) {
    infiniopScaledMmW8A8Descriptor_t descriptor = nullptr;
    INFINICORE_CHECK_ERROR(infiniopCreateScaledMmW8A8Descriptor(
        context::getInfiniopHandle(out->device()),
        &descriptor,
        out->desc(),
        a->desc(),
        b->desc(),
        a_scales->desc(),
        b_scales->desc(),
        bias ? (*bias)->desc() : nullptr,
        trans_weight));
    const auto status = infiniopScaledMmW8A8(
        descriptor,
        out->data(),
        a->data(),
        b->data(),
        a_scales->data(),
        b_scales->data(),
        bias ? (*bias)->data() : nullptr,
        context::getStream());
    const auto destroy_status = infiniopDestroyScaledMmW8A8Descriptor(descriptor);
    INFINICORE_CHECK_ERROR(status);
    INFINICORE_CHECK_ERROR(destroy_status);
}

static bool registered = []() {
    vendor_ops::scaled_mm_w8a8_dispatcher().registerDevice(Device::Type::HYGON, &run);
    return true;
}();

} // namespace infinicore::op::scaled_mm_w8a8_impl::hygon
#endif
