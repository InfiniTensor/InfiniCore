#if defined(ENABLE_HYGON_API)
#include "../../infiniop_impl.hpp"
#include "../../vendor_ops/vendor_ops_dispatch.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/ops/concat_mla_q.hpp"
#include "infiniop/ops/concat_mla_q.h"

namespace infinicore::op::concat_mla_q_impl::hygon {

void run(const Tensor &ql_nope, const Tensor &q_pe, Tensor q_out) {
    INFINICORE_CHECK_ERROR(infiniopConcatMlaQ(
        context::getInfiniopHandle(q_out->device()),
        ql_nope->desc(),
        q_pe->desc(),
        q_out->desc(),
        ql_nope->data(),
        q_pe->data(),
        q_out->data(),
        context::getStream()));
}

static bool registered = []() {
    vendor_ops::concat_mla_q_dispatcher().registerDevice(Device::Type::HYGON, &run);
    return true;
}();

} // namespace infinicore::op::concat_mla_q_impl::hygon
#endif
