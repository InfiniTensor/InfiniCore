#if defined(ENABLE_ILUVATAR_VENDOR_OPS)
#include "../../vendor_ops/vendor_ops_dispatch.hpp"

#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/adaptor/iluvatar_vendor_adaptor.hpp"

#include <stdexcept>

namespace infinicore::op::concat_mla_q_impl::iluvatar {

void run(const Tensor &ql_nope, const Tensor &q_pe, Tensor q_out) {
    if (ql_nope->size(2) != 512 || q_pe->size(2) != 64 || q_out->size(2) != 576) {
        throw std::runtime_error("concat_mla_q Iluvatar implementation supports only GLM MLA dims 512 + 64 -> 576");
    }
    if (!adaptor::iluvatar_vendor::concat_mla_q_available()) {
        throw std::runtime_error("concat_mla_q requires the Iluvatar vendor extension");
    }
    auto ql_nope_at = adaptor::to_aten_tensor(ql_nope);
    auto q_pe_at = adaptor::to_aten_tensor(q_pe);
    auto q_out_at = adaptor::to_aten_tensor(q_out);
    adaptor::iluvatar_vendor::concat_mla_q(
        ql_nope_at, q_pe_at,
        q_out_at);
}

static bool registered = []() {
    vendor_ops::concat_mla_q_dispatcher().registerDevice(Device::Type::ILUVATAR, &run);
    return true;
}();

} // namespace infinicore::op::concat_mla_q_impl::iluvatar
#endif
