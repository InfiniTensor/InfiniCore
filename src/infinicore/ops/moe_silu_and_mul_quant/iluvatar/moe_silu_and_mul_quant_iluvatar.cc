#if defined(ENABLE_ILUVATAR_VENDOR_OPS)
#include "../../vendor_ops/vendor_ops_dispatch.hpp"

#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/adaptor/iluvatar_vendor_adaptor.hpp"

#include <stdexcept>

namespace infinicore::op::moe_silu_and_mul_quant_impl::iluvatar {

void run(Tensor output,
         std::optional<Tensor> output_scale,
         const Tensor &input,
         int64_t format) {
    if (!adaptor::iluvatar_vendor::silu_and_mul_quant_available()) {
        throw std::runtime_error("moe_silu_and_mul_quant requires the Iluvatar vendor extension");
    }
    std::optional<at::Tensor> scale_at;
    if (output_scale) {
        scale_at = adaptor::to_aten_tensor(*output_scale);
    }
    auto output_at = adaptor::to_aten_tensor(output);
    auto input_at = adaptor::to_aten_tensor(input);
    adaptor::iluvatar_vendor::silu_and_mul_quant(
        output_at, scale_at, input_at, format);
}

static bool registered = []() {
    vendor_ops::moe_silu_and_mul_quant_dispatcher().registerDevice(Device::Type::ILUVATAR, &run);
    return true;
}();

} // namespace infinicore::op::moe_silu_and_mul_quant_impl::iluvatar
#endif
