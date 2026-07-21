#include "infinicore/ops/mul.hpp"
#include "../../utils.hpp"

#include <cstdlib>
#include <string>

namespace infinicore::op {

namespace {

/// Diagnose/fix: MetaX GraphLaunch ATUs when ``Mul`` is captured after
/// PagedAttention (binary-search: MAX_OPS through PagedAttn PASS; +Mul FAIL;
/// host-break attn then single-Mul segment also FAIL). Opt-in host-break.
bool mul_host_break_enabled() {
    const char *v = std::getenv("INFINI_MUL_HOST_BREAK");
    return v != nullptr && v[0] != '\0' && std::string(v) != "0";
}

} // namespace

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Mul);

Mul::Mul(Tensor c, const Tensor &a, const Tensor &b) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(c, a, b);
    INFINICORE_GRAPH_OP_DISPATCH(c->device().getType(), c, a, b);
    host_break_ = mul_host_break_enabled();
}

void Mul::execute(Tensor c, const Tensor &a, const Tensor &b) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Mul, c, a, b);
}

Tensor mul(const Tensor &a, const Tensor &b) {
    auto c = Tensor::empty(a->shape(), a->dtype(), a->device());
    mul_(c, a, b);
    return c;
}

void mul_(Tensor c, const Tensor &a, const Tensor &b) {
    Mul::execute(c, a, b);
}

} // namespace infinicore::op
