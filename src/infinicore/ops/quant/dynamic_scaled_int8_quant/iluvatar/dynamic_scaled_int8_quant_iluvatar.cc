#if defined(ENABLE_ILUVATAR_VENDOR_OPS)
#include "../../../vendor_ops/vendor_ops_dispatch.hpp"

#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/adaptor/iluvatar_vendor_adaptor.hpp"

#include <stdexcept>

namespace infinicore::op::dynamic_scaled_int8_quant_impl::iluvatar {

void run(Tensor output, const Tensor &input, Tensor input_scales) {
    if (!adaptor::iluvatar_vendor::dynamic_scaled_int8_quant_available()) {
        throw std::runtime_error("dynamic_scaled_int8_quant requires the Iluvatar vendor extension");
    }
    auto output_at = adaptor::to_aten_tensor(output);
    auto input_scales_at = adaptor::to_aten_tensor(input_scales);
    auto input_at = adaptor::to_aten_tensor(input);
    adaptor::iluvatar_vendor::dynamic_scaled_int8_quant(
        output_at, input_scales_at,
        input_at);
}

static bool registered = []() {
    vendor_ops::dynamic_scaled_int8_quant_dispatcher().registerDevice(Device::Type::ILUVATAR, &run);
    return true;
}();

} // namespace infinicore::op::dynamic_scaled_int8_quant_impl::iluvatar
#endif
