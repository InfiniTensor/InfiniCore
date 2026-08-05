#if defined(ENABLE_ILUVATAR_VENDOR_OPS)
#include "../../vendor_ops/vendor_ops_dispatch.hpp"

#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/adaptor/iluvatar_vendor_adaptor.hpp"

#include <stdexcept>

namespace infinicore::op::scaled_mm_w4a8_impl::iluvatar {

void run(Tensor out,
         const Tensor &a,
         const Tensor &b,
         const Tensor &a_scales,
         const Tensor &b_scales,
         std::optional<Tensor> bias,
         bool trans_weight) {
    if (!adaptor::iluvatar_vendor::scaled_mm_w4a8_available()) {
        throw std::runtime_error("scaled_mm_w4a8 requires the Iluvatar vendor extension");
    }
    std::optional<at::Tensor> bias_at;
    if (bias) {
        bias_at = adaptor::to_aten_tensor(*bias);
    }
    auto out_at = adaptor::to_aten_tensor(out);
    auto a_at = adaptor::to_aten_tensor(a);
    auto b_at = adaptor::to_aten_tensor(b);
    auto a_scales_at = adaptor::to_aten_tensor(a_scales);
    auto b_scales_at = adaptor::to_aten_tensor(b_scales);
    adaptor::iluvatar_vendor::scaled_mm_w4a8(
        out_at, a_at,
        b_at, a_scales_at,
        b_scales_at, bias_at, trans_weight);
}

static bool registered = []() {
    vendor_ops::scaled_mm_w4a8_dispatcher().registerDevice(Device::Type::ILUVATAR, &run);
    return true;
}();

} // namespace infinicore::op::scaled_mm_w4a8_impl::iluvatar
#endif
