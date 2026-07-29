#include "infinicore/ops/ernie45_rope.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Ernie45MRoPE);
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Ernie45VisionRoPE);

Ernie45MRoPE::Ernie45MRoPE(Tensor q,
                           Tensor k,
                           const Tensor &positions,
                           double rope_theta,
                           size_t section_h,
                           size_t section_w,
                           size_t section_t) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(q, k, positions);
    INFINICORE_GRAPH_OP_DISPATCH(q->device().getType(), q, k, positions, rope_theta, section_h, section_w, section_t);
}

void Ernie45MRoPE::execute(Tensor q,
                           Tensor k,
                           const Tensor &positions,
                           double rope_theta,
                           size_t section_h,
                           size_t section_w,
                           size_t section_t) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Ernie45MRoPE, q, k, positions, rope_theta, section_h, section_w, section_t);
}

Tensor ernie45_mrope_(Tensor q,
                      Tensor k,
                      const Tensor &positions,
                      double rope_theta,
                      size_t section_h,
                      size_t section_w,
                      size_t section_t) {
    Ernie45MRoPE::execute(q, k, positions, rope_theta, section_h, section_w, section_t);
    return q;
}

Ernie45VisionRoPE::Ernie45VisionRoPE(Tensor q,
                                     Tensor k,
                                     const Tensor &positions,
                                     double rope_theta) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(q, k, positions);
    INFINICORE_GRAPH_OP_DISPATCH(q->device().getType(), q, k, positions, rope_theta);
}

void Ernie45VisionRoPE::execute(Tensor q,
                                Tensor k,
                                const Tensor &positions,
                                double rope_theta) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Ernie45VisionRoPE, q, k, positions, rope_theta);
}

Tensor ernie45_vision_rope_(Tensor q,
                            Tensor k,
                            const Tensor &positions,
                            double rope_theta) {
    Ernie45VisionRoPE::execute(q, k, positions, rope_theta);
    return q;
}

} // namespace infinicore::op
