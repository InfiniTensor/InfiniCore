#include "infinicore/ops/ones.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Ones);

Ones::Ones(Tensor output) {
    INFINICORE_GRAPH_OP_DISPATCH(output->device().getType(), output);
}

void Ones::execute(Tensor output) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Ones, output);
}

void ones_(Tensor output) {
    Ones::execute(output);
}

} // namespace infinicore::op
