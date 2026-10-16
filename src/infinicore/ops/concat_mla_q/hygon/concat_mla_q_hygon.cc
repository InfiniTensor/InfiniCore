#if defined(ENABLE_HYGON_API)

#include "../../vendor_ops/vendor_ops_dispatch.hpp"

namespace infinicore::op::concat_mla_q_impl::hygon {

void run(const Tensor &ql_nope, const Tensor &q_pe, Tensor q_out) {
    const size_t nope_dim = ql_nope->size(2);
    q_out->narrow({{2, 0, nope_dim}})->copy_from(ql_nope);
    q_out->narrow({{2, nope_dim, q_pe->size(2)}})->copy_from(q_pe);
}

static bool registered = []() {
    vendor_ops::concat_mla_q_dispatcher().registerDevice(
        Device::Type::HYGON, &run);
    return true;
}();

} // namespace infinicore::op::concat_mla_q_impl::hygon

#endif
