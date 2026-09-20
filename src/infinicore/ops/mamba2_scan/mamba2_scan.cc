#include "infinicore/ops/mamba2_scan.hpp"
#include "../../utils.hpp"

namespace infinicore::op {
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Mamba2Scan);
Mamba2Scan::Mamba2Scan(Tensor out, const Tensor &x, const Tensor &dt, const Tensor &b, const Tensor &c, const Tensor &a, const Tensor &d, const Tensor &dt_bias, Tensor state, const Tensor &offsets, const Tensor &initial_indices, const Tensor &final_indices) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, x, dt, b, c, a, d, dt_bias, state, offsets, initial_indices, final_indices);
    INFINICORE_GRAPH_OP_DISPATCH(out->device().getType(), out, x, dt, b, c, a, d, dt_bias, state, offsets, initial_indices, final_indices);
}
void Mamba2Scan::execute(Tensor out, const Tensor &x, const Tensor &dt, const Tensor &b, const Tensor &c, const Tensor &a, const Tensor &d, const Tensor &dt_bias, Tensor state, const Tensor &offsets, const Tensor &initial_indices, const Tensor &final_indices) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Mamba2Scan, out, x, dt, b, c, a, d, dt_bias, state, offsets, initial_indices, final_indices);
}
Tensor mamba2_scan(const Tensor &x, const Tensor &dt, const Tensor &b, const Tensor &c, const Tensor &a, const Tensor &d, const Tensor &dt_bias, Tensor state, const Tensor &offsets, const Tensor &initial_indices, const Tensor &final_indices) {
    auto out = Tensor::empty(x->shape(), x->dtype(), x->device());
    mamba2_scan_(out, x, dt, b, c, a, d, dt_bias, state, offsets, initial_indices, final_indices);
    return out;
}
void mamba2_scan_(Tensor out, const Tensor &x, const Tensor &dt, const Tensor &b, const Tensor &c, const Tensor &a, const Tensor &d, const Tensor &dt_bias, Tensor state, const Tensor &offsets, const Tensor &initial_indices, const Tensor &final_indices) {
    Mamba2Scan::execute(out, x, dt, b, c, a, d, dt_bias, state, offsets, initial_indices, final_indices);
}
} // namespace infinicore::op
