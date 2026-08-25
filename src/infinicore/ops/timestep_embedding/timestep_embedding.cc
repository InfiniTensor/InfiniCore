#include "infinicore/ops/timestep_embedding.hpp"

#include "../../utils.hpp"

#include <stdexcept>

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(TimestepEmbedding);

TimestepEmbedding::TimestepEmbedding(Tensor output,
                                     const Tensor &timestep,
                                     float max_period) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, timestep);
    INFINICORE_GRAPH_OP_DISPATCH(
        output->device().getType(), output, timestep, max_period);
}

void TimestepEmbedding::execute(Tensor output,
                                const Tensor &timestep,
                                float max_period) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(
        TimestepEmbedding, output, timestep, max_period);
}

Tensor timestep_embedding(const Tensor &timestep,
                          size_t embedding_dim,
                          float max_period) {
    auto output = Tensor::empty(
        {timestep->numel(), embedding_dim},
        DataType::F32,
        timestep->device());
    timestep_embedding_(output, timestep, max_period);
    return output;
}

void timestep_embedding_(Tensor output,
                         const Tensor &timestep,
                         float max_period) {
    if (timestep->ndim() != 1) {
        throw std::runtime_error("timestep_embedding expects timestep shape [N]");
    }
    if (output->ndim() != 2 || output->size(0) != timestep->size(0)
        || output->size(1) == 0 || output->size(1) % 2 != 0) {
        throw std::runtime_error(
            "timestep_embedding expects output shape [N, even embedding_dim]");
    }
    if (output->dtype() != DataType::F32) {
        throw std::runtime_error("timestep_embedding output must be float32");
    }
    if (max_period <= 0.0f) {
        throw std::runtime_error("timestep_embedding max_period must be positive");
    }
    TimestepEmbedding::execute(output, timestep, max_period);
}

} // namespace infinicore::op
