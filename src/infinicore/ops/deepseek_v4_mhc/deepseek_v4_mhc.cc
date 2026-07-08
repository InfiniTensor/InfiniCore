#include "infinicore/ops/deepseek_v4_mhc.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(DeepseekV4MHCParams);
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(DeepseekV4MHCPreCollapse);
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(DeepseekV4MHCScaleMixes);

DeepseekV4MHCParams::DeepseekV4MHCParams(Tensor pre,
                                         Tensor post,
                                         Tensor comb,
                                         const Tensor &mixes,
                                         const Tensor &base,
                                         const Tensor &scale,
                                         size_t sinkhorn_iters,
                                         float epsilon) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(pre, post, comb, mixes, base, scale);
    INFINICORE_GRAPH_OP_DISPATCH(
        pre->device().getType(),
        pre,
        post,
        comb,
        mixes,
        base,
        scale,
        sinkhorn_iters,
        epsilon);
}

void DeepseekV4MHCParams::execute(Tensor pre,
                                  Tensor post,
                                  Tensor comb,
                                  const Tensor &mixes,
                                  const Tensor &base,
                                  const Tensor &scale,
                                  size_t sinkhorn_iters,
                                  float epsilon) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(
        DeepseekV4MHCParams,
        pre,
        post,
        comb,
        mixes,
        base,
        scale,
        sinkhorn_iters,
        epsilon);
}

std::tuple<Tensor, Tensor, Tensor> deepseek_v4_mhc_params(
    const Tensor &mixes,
    const Tensor &base,
    const Tensor &scale,
    size_t sinkhorn_iters,
    float epsilon) {
    const auto &shape = mixes->shape();
    INFINICORE_ASSERT(shape.size() == 3);
    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];
    const size_t mix_hc = shape[2];
    size_t hc_mult = 0;
    for (size_t h = 1; h <= 16; ++h) {
        if ((2 + h) * h == mix_hc) {
            hc_mult = h;
            break;
        }
    }
    INFINICORE_ASSERT(hc_mult != 0);
    auto pre = Tensor::empty({batch_size, seq_len, hc_mult}, mixes->dtype(), mixes->device());
    auto post = Tensor::empty({batch_size, seq_len, hc_mult}, mixes->dtype(), mixes->device());
    auto comb = Tensor::empty({batch_size, seq_len, hc_mult, hc_mult}, mixes->dtype(), mixes->device());
    deepseek_v4_mhc_params_(pre, post, comb, mixes, base, scale, sinkhorn_iters, epsilon);
    return {pre, post, comb};
}

void deepseek_v4_mhc_params_(Tensor pre,
                             Tensor post,
                             Tensor comb,
                             const Tensor &mixes,
                             const Tensor &base,
                             const Tensor &scale,
                             size_t sinkhorn_iters,
                             float epsilon) {
    DeepseekV4MHCParams::execute(pre, post, comb, mixes, base, scale, sinkhorn_iters, epsilon);
}


DeepseekV4MHCPreCollapse::DeepseekV4MHCPreCollapse(Tensor collapsed,
                                                   Tensor post,
                                                   Tensor comb,
                                                   const Tensor &x,
                                                   const Tensor &mixes,
                                                   const Tensor &base,
                                                   const Tensor &scale,
                                                   size_t sinkhorn_iters,
                                                   float epsilon) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(collapsed, post, comb, x, mixes, base, scale);
    INFINICORE_GRAPH_OP_DISPATCH(
        collapsed->device().getType(),
        collapsed,
        post,
        comb,
        x,
        mixes,
        base,
        scale,
        sinkhorn_iters,
        epsilon);
}

void DeepseekV4MHCPreCollapse::execute(Tensor collapsed,
                                       Tensor post,
                                       Tensor comb,
                                       const Tensor &x,
                                       const Tensor &mixes,
                                       const Tensor &base,
                                       const Tensor &scale,
                                       size_t sinkhorn_iters,
                                       float epsilon) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(
        DeepseekV4MHCPreCollapse,
        collapsed,
        post,
        comb,
        x,
        mixes,
        base,
        scale,
        sinkhorn_iters,
        epsilon);
}

std::tuple<Tensor, Tensor, Tensor> deepseek_v4_mhc_pre_collapse(
    const Tensor &x,
    const Tensor &mixes,
    const Tensor &base,
    const Tensor &scale,
    size_t sinkhorn_iters,
    float epsilon) {
    const auto &x_shape = x->shape();
    INFINICORE_ASSERT(x_shape.size() == 4);
    auto collapsed = Tensor::empty({x_shape[0], x_shape[1], x_shape[3]}, x->dtype(), x->device());
    auto post = Tensor::empty({x_shape[0], x_shape[1], x_shape[2]}, mixes->dtype(), mixes->device());
    auto comb = Tensor::empty({x_shape[0], x_shape[1], x_shape[2], x_shape[2]}, mixes->dtype(), mixes->device());
    deepseek_v4_mhc_pre_collapse_(collapsed, post, comb, x, mixes, base, scale, sinkhorn_iters, epsilon);
    return {collapsed, post, comb};
}

void deepseek_v4_mhc_pre_collapse_(
    Tensor collapsed,
    Tensor post,
    Tensor comb,
    const Tensor &x,
    const Tensor &mixes,
    const Tensor &base,
    const Tensor &scale,
    size_t sinkhorn_iters,
    float epsilon) {
    DeepseekV4MHCPreCollapse::execute(collapsed, post, comb, x, mixes, base, scale, sinkhorn_iters, epsilon);
}


DeepseekV4MHCScaleMixes::DeepseekV4MHCScaleMixes(Tensor scaled,
                                                 const Tensor &x,
                                                 const Tensor &raw_mixes,
                                                 float epsilon) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(scaled, x, raw_mixes);
    INFINICORE_GRAPH_OP_DISPATCH(
        scaled->device().getType(),
        scaled,
        x,
        raw_mixes,
        epsilon);
}

void DeepseekV4MHCScaleMixes::execute(Tensor scaled,
                                      const Tensor &x,
                                      const Tensor &raw_mixes,
                                      float epsilon) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(
        DeepseekV4MHCScaleMixes,
        scaled,
        x,
        raw_mixes,
        epsilon);
}

Tensor deepseek_v4_mhc_scale_mixes(
    const Tensor &x,
    const Tensor &raw_mixes,
    float epsilon) {
    auto scaled = Tensor::empty(raw_mixes->shape(), raw_mixes->dtype(), raw_mixes->device());
    deepseek_v4_mhc_scale_mixes_(scaled, x, raw_mixes, epsilon);
    return scaled;
}

void deepseek_v4_mhc_scale_mixes_(
    Tensor scaled,
    const Tensor &x,
    const Tensor &raw_mixes,
    float epsilon) {
    DeepseekV4MHCScaleMixes::execute(scaled, x, raw_mixes, epsilon);
}

} // namespace infinicore::op
