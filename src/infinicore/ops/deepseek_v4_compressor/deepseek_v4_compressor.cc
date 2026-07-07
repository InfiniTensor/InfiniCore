#include "infinicore/ops/deepseek_v4_compressor.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(DeepseekV4Compressor);

DeepseekV4Compressor::DeepseekV4Compressor(Tensor out,
                                           const Tensor &kv,
                                           const Tensor &score,
                                           const Tensor &ape,
                                           const Tensor &norm_weight,
                                           size_t compress_ratio,
                                           float epsilon) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, kv, score, ape, norm_weight);
    INFINICORE_GRAPH_OP_DISPATCH(
        out->device().getType(),
        out,
        kv,
        score,
        ape,
        norm_weight,
        compress_ratio,
        epsilon);
}

void DeepseekV4Compressor::execute(Tensor out,
                                   const Tensor &kv,
                                   const Tensor &score,
                                   const Tensor &ape,
                                   const Tensor &norm_weight,
                                   size_t compress_ratio,
                                   float epsilon) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(
        DeepseekV4Compressor,
        out,
        kv,
        score,
        ape,
        norm_weight,
        compress_ratio,
        epsilon);
}

Tensor deepseek_v4_compressor(const Tensor &kv,
                              const Tensor &score,
                              const Tensor &ape,
                              const Tensor &norm_weight,
                              size_t compress_ratio,
                              float epsilon) {
    const auto &kv_shape = kv->shape();
    const auto &w_shape = norm_weight->shape();
    INFINICORE_ASSERT(kv_shape.size() == 3);
    INFINICORE_ASSERT(w_shape.size() == 1);
    INFINICORE_ASSERT(compress_ratio > 0);
    const size_t num_blocks = kv_shape[1] / compress_ratio;
    auto out = Tensor::empty({kv_shape[0], num_blocks, w_shape[0]}, kv->dtype(), kv->device());
    deepseek_v4_compressor_(out, kv, score, ape, norm_weight, compress_ratio, epsilon);
    return out;
}

void deepseek_v4_compressor_(Tensor out,
                             const Tensor &kv,
                             const Tensor &score,
                             const Tensor &ape,
                             const Tensor &norm_weight,
                             size_t compress_ratio,
                             float epsilon) {
    DeepseekV4Compressor::execute(out, kv, score, ape, norm_weight, compress_ratio, epsilon);
}

} // namespace infinicore::op
