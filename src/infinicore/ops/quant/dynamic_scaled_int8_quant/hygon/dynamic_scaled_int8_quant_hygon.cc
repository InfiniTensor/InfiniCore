#if defined(ENABLE_ATEN) && defined(ENABLE_HYGON_API)

#include "../../../vendor_ops/vendor_ops_dispatch.hpp"

#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/adaptor/hygon_aiter_adaptor.hpp"

#include <stdexcept>

namespace infinicore::op::dynamic_scaled_int8_quant_impl::hygon {

void run(Tensor output, const Tensor &input, Tensor input_scales) {
    if (!adaptor::hygon_aiter::dynamic_per_token_scaled_quant_available()) {
        throw std::runtime_error(
            "dynamic_scaled_int8_quant requires the AITER Hygon extension");
    }
    adaptor::set_aten_stream_to_infinicore();
    auto output_at = adaptor::to_aten_tensor(output, at::kFloat8_e4m3fn);
    auto input_at = adaptor::to_aten_tensor(input);
    auto input_scales_at = adaptor::to_aten_tensor(input_scales);
    adaptor::hygon_aiter::dynamic_per_token_scaled_quant(
        output_at, input_at, input_scales_at);
}

static bool registered = []() {
    vendor_ops::dynamic_scaled_int8_quant_dispatcher().registerDevice(
        Device::Type::HYGON, &run);
    return true;
}();

} // namespace infinicore::op::dynamic_scaled_int8_quant_impl::hygon

#endif
