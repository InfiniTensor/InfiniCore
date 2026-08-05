#if defined(ENABLE_ILUVATAR_VENDOR_OPS)
#include "../../vendor_ops/vendor_ops_dispatch.hpp"

#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/adaptor/iluvatar_vendor_adaptor.hpp"

#include <stdexcept>

namespace infinicore::op::add_rms_norm_impl::iluvatar {

void run(Tensor input, Tensor residual, const Tensor &weight, float epsilon) {
    if (!adaptor::iluvatar_vendor::available()) {
        throw std::runtime_error("add_rms_norm_inplace requires the Iluvatar vendor extension");
    }
    auto input_at = adaptor::to_aten_tensor(input);
    auto residual_at = adaptor::to_aten_tensor(residual);
    auto weight_at = adaptor::to_aten_tensor(weight);
    adaptor::iluvatar_vendor::fused_add_rms_norm(input_at, residual_at, weight_at, epsilon);
}

static bool registered = []() {
    vendor_ops::add_rms_norm_inplace_dispatcher().registerDevice(Device::Type::ILUVATAR, &run);
    return true;
}();

} // namespace infinicore::op::add_rms_norm_impl::iluvatar
#endif
