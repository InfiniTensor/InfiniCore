#include "infinicore/ops/deepseek_v4_mhc_head.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(DeepseekV4MHCHeadCollapse);

DeepseekV4MHCHeadCollapse::DeepseekV4MHCHeadCollapse(Tensor y,
                                                     const Tensor &x,
                                                     const Tensor &mixes,
                                                     const Tensor &base,
                                                     const Tensor &scale,
                                                     float epsilon) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(y, x, mixes, base, scale);
    INFINICORE_GRAPH_OP_DISPATCH(
        y->device().getType(),
        y,
        x,
        mixes,
        base,
        scale,
        epsilon);
}

void DeepseekV4MHCHeadCollapse::execute(Tensor y,
                                        const Tensor &x,
                                        const Tensor &mixes,
                                        const Tensor &base,
                                        const Tensor &scale,
                                        float epsilon) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(
        DeepseekV4MHCHeadCollapse,
        y,
        x,
        mixes,
        base,
        scale,
        epsilon);
}

Tensor deepseek_v4_mhc_head_collapse(
    const Tensor &x,
    const Tensor &mixes,
    const Tensor &base,
    const Tensor &scale,
    float epsilon) {
    const auto &x_shape = x->shape();
    INFINICORE_ASSERT(x_shape.size() == 4);
    auto y = Tensor::empty({x_shape[0], x_shape[1], x_shape[3]}, x->dtype(), x->device());
    deepseek_v4_mhc_head_collapse_(y, x, mixes, base, scale, epsilon);
    return y;
}

void deepseek_v4_mhc_head_collapse_(
    Tensor y,
    const Tensor &x,
    const Tensor &mixes,
    const Tensor &base,
    const Tensor &scale,
    float epsilon) {
    DeepseekV4MHCHeadCollapse::execute(y, x, mixes, base, scale, epsilon);
}

} // namespace infinicore::op
