#include "infinicore/ops/deepseek_v4_mhc.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(DeepseekV4MHCParams);

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

} // namespace infinicore::op
