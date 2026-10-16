#if defined(ENABLE_ATEN) && defined(ENABLE_HYGON_API)

#include "infinicore/ops/fp8_e4m3_dequant.hpp"

#include "infinicore/adaptor/aten_adaptor.hpp"

namespace infinicore::op::fp8_e4m3_dequant_impl::hygon {

void run(Tensor output, const Tensor &input, const Tensor &scales) {
    adaptor::set_aten_stream_to_infinicore();
    auto output_at = adaptor::to_aten_tensor(output);
    auto input_at = adaptor::to_aten_tensor(input, at::kFloat8_e4m3fn);
    auto scales_at = adaptor::to_aten_tensor(scales);
    output_at.copy_(input_at.to(adaptor::to_at_dtype(output->dtype())));
    output_at.mul_(scales_at);
}

static bool registered = []() {
    Fp8E4m3Dequant::dispatcher().registerDevice(Device::Type::HYGON, &run);
    return true;
}();

} // namespace infinicore::op::fp8_e4m3_dequant_impl::hygon

#endif
