#if defined(ENABLE_HYGON_API)
#include "../../vendor_ops/vendor_ops_dispatch.hpp"

#include "infinicore/ops/dynamic_scaled_int8_quant.hpp"
#include "infinicore/ops/silu_and_mul.hpp"

#include <stdexcept>

namespace infinicore::op::moe_silu_and_mul_quant_impl::hygon {

void run(Tensor output,
         std::optional<Tensor> output_scale,
         const Tensor &input,
         int64_t format) {
    if (format == 0) {
        silu_and_mul_(output, input);
        return;
    }
    if (format != 1 || !output_scale) {
        throw std::runtime_error(
            "Hygon moe_silu_and_mul_quant supports normal or conventional INT8 format");
    }
    auto activated = Tensor::empty(
        output->shape(), input->dtype(), input->device());
    silu_and_mul_(activated, input);
    dynamic_scaled_int8_quant_(output, activated, *output_scale);
}

static bool registered = []() {
    vendor_ops::moe_silu_and_mul_quant_dispatcher().registerDevice(
        Device::Type::HYGON, &run);
    return true;
}();

} // namespace infinicore::op::moe_silu_and_mul_quant_impl::hygon
#endif
