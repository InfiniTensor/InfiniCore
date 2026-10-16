#if defined(ENABLE_ATEN) && defined(ENABLE_HYGON_API)

#include "../../vendor_ops/vendor_ops_dispatch.hpp"

#include "infinicore/adaptor/aten_adaptor.hpp"

#include <ATen/ops/_scaled_mm.h>

#include <stdexcept>

namespace infinicore::op::scaled_mm_w8a8_impl::hygon {

void run(Tensor out,
         const Tensor &a,
         const Tensor &b,
         const Tensor &a_scales,
         const Tensor &b_scales,
         std::optional<Tensor> bias,
         bool trans_weight) {
    if (!trans_weight) {
        throw std::runtime_error(
            "Hygon scaled_mm_w8a8 expects row-major [out,in] weights");
    }

    adaptor::set_aten_stream_to_infinicore();
    auto out_at = adaptor::to_aten_tensor(out);
    auto a_at = adaptor::to_aten_tensor(a, at::kFloat8_e4m3fn);
    auto b_at = adaptor::to_aten_tensor(b, at::kFloat8_e4m3fn);
    auto a_scales_at = adaptor::to_aten_tensor(a_scales);
    auto b_scales_at = adaptor::to_aten_tensor(b_scales);
    std::optional<at::Tensor> bias_at;
    if (bias.has_value()) {
        bias_at = adaptor::to_aten_tensor(*bias);
    }

    at::_scaled_mm_out(
        out_at,
        a_at,
        b_at.transpose(0, 1),
        a_scales_at,
        b_scales_at.transpose(0, 1),
        bias_at,
        std::nullopt,
        adaptor::to_at_dtype(out->dtype()),
        false);
}

static bool registered = []() {
    vendor_ops::scaled_mm_w8a8_dispatcher().registerDevice(
        Device::Type::HYGON, &run);
    return true;
}();

} // namespace infinicore::op::scaled_mm_w8a8_impl::hygon

#endif
