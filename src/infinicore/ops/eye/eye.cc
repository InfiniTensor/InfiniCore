#include "infinicore/ops/eye.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Eye);

Eye::Eye(Tensor y) {
    INFINICORE_GRAPH_OP_DISPATCH(y->device().getType(), y);
}

void Eye::execute(Tensor y) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Eye, y);
}

Tensor eye(size_t n, std::optional<size_t> m, const DataType &dtype, const Device &device) {
    size_t m_val = m.value_or(n);
    auto y = Tensor::empty({n, m_val}, dtype, device);
    Eye::execute(y);
    return y;
}

} // namespace infinicore::op
