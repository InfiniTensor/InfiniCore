#include "infinicore/ops/rms_rotary_embedding.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(RMSRotaryEmbedding);

RMSRotaryEmbedding::RMSRotaryEmbedding(Tensor query,
                                       Tensor key,
                                       const Tensor &positions,
                                       int64_t head_size,
                                       const Tensor &cos_sin_cache,
                                       bool is_neox,
                                       const Tensor &q_weight,
                                       const Tensor &k_weight,
                                       float epsilon) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(query, key, positions, cos_sin_cache, q_weight, k_weight);
    INFINICORE_GRAPH_OP_DISPATCH(query->device().getType(),
                                 query,
                                 key,
                                 positions,
                                 head_size,
                                 cos_sin_cache,
                                 is_neox,
                                 q_weight,
                                 k_weight,
                                 epsilon);
}

void RMSRotaryEmbedding::execute(Tensor query,
                                 Tensor key,
                                 const Tensor &positions,
                                 int64_t head_size,
                                 const Tensor &cos_sin_cache,
                                 bool is_neox,
                                 const Tensor &q_weight,
                                 const Tensor &k_weight,
                                 float epsilon) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(RMSRotaryEmbedding,
                                      query,
                                      key,
                                      positions,
                                      head_size,
                                      cos_sin_cache,
                                      is_neox,
                                      q_weight,
                                      k_weight,
                                      epsilon);
}

bool rms_rotary_embedding_fuse_available(const Device &device) {
    return RMSRotaryEmbedding::plan_dispatcher().lookup(device.getType()) != nullptr;
}

void rms_rotary_embedding_fuse_(Tensor query,
                                Tensor key,
                                const Tensor &positions,
                                int64_t head_size,
                                const Tensor &cos_sin_cache,
                                bool is_neox,
                                const Tensor &q_weight,
                                const Tensor &k_weight,
                                float epsilon) {
    RMSRotaryEmbedding::execute(query,
                                key,
                                positions,
                                head_size,
                                cos_sin_cache,
                                is_neox,
                                q_weight,
                                k_weight,
                                epsilon);
}

} // namespace infinicore::op
