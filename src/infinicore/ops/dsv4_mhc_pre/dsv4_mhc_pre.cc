#include "infinicore/ops/dsv4_mhc_pre.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Dsv4MhcPre);

Dsv4MhcPre::Dsv4MhcPre(Tensor output, const Tensor &input, const Tensor &scale, const Tensor &base, float eps) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input, scale, base);
    INFINICORE_GRAPH_OP_DISPATCH(output->device().getType(), output, input, scale, base, eps);
}

void Dsv4MhcPre::execute(Tensor output, const Tensor &input, const Tensor &scale, const Tensor &base, float eps) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Dsv4MhcPre, output, input, scale, base, eps);
}

void dsv4_mhc_pre_(Tensor output, const Tensor &input, const Tensor &scale, const Tensor &base, float eps) {
    Dsv4MhcPre::execute(output, input, scale, base, eps);
}

} // namespace infinicore::op
