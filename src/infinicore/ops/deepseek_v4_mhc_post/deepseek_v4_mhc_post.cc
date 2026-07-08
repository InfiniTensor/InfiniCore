#include "infinicore/ops/deepseek_v4_mhc_post.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(DeepseekV4MHCPost);

DeepseekV4MHCPost::DeepseekV4MHCPost(Tensor y,
                                     const Tensor &new_x,
                                     const Tensor &residual,
                                     const Tensor &post,
                                     const Tensor &comb) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(y, new_x, residual, post, comb);
    INFINICORE_GRAPH_OP_DISPATCH(
        y->device().getType(),
        y,
        new_x,
        residual,
        post,
        comb);
}

void DeepseekV4MHCPost::execute(Tensor y,
                                const Tensor &new_x,
                                const Tensor &residual,
                                const Tensor &post,
                                const Tensor &comb) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(
        DeepseekV4MHCPost,
        y,
        new_x,
        residual,
        post,
        comb);
}

Tensor deepseek_v4_mhc_post(
    const Tensor &new_x,
    const Tensor &residual,
    const Tensor &post,
    const Tensor &comb) {
    const auto &residual_shape = residual->shape();
    INFINICORE_ASSERT(residual_shape.size() == 4);
    auto y = Tensor::empty(residual_shape, new_x->dtype(), new_x->device());
    deepseek_v4_mhc_post_(y, new_x, residual, post, comb);
    return y;
}

void deepseek_v4_mhc_post_(
    Tensor y,
    const Tensor &new_x,
    const Tensor &residual,
    const Tensor &post,
    const Tensor &comb) {
    DeepseekV4MHCPost::execute(y, new_x, residual, post, comb);
}

} // namespace infinicore::op
