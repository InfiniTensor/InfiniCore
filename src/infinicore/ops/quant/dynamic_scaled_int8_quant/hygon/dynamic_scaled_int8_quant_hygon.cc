#if defined(ENABLE_HYGON_API)
#include "../../../vendor_ops/vendor_ops_dispatch.hpp"

#include "infinicore/ops/per_channel_quant_i8.hpp"

namespace infinicore::op::dynamic_scaled_int8_quant_impl::hygon {

void run(Tensor output, const Tensor &input, Tensor input_scales) {
    per_channel_quant_i8_(input, output, input_scales);
}

static bool registered = []() {
    vendor_ops::dynamic_scaled_int8_quant_dispatcher().registerDevice(
        Device::Type::HYGON, &run);
    return true;
}();

} // namespace infinicore::op::dynamic_scaled_int8_quant_impl::hygon
#endif
