#include "infinicore/ops/zeros.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Zeros);

Zeros::Zeros(Tensor output) {
    INFINICORE_GRAPH_OP_DISPATCH(output->device().getType(), output);
}

void Zeros::execute(Tensor output) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Zeros, output);
}

void zeros_(Tensor output) {
    Zeros::execute(output);
}

} // namespace infinicore::op
