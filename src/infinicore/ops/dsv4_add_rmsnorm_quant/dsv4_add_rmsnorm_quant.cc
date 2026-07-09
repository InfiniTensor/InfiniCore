#include "infinicore/ops/dsv4_add_rmsnorm_quant.hpp"
#include "../../utils.hpp"
namespace infinicore::op {
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Dsv4AddRMSNormQuant);
Dsv4AddRMSNormQuant::Dsv4AddRMSNormQuant(Tensor res, Tensor q, Tensor scale, const Tensor &x, const Tensor &weight, float epsilon) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(res, q, scale, x, weight);
    INFINICORE_GRAPH_OP_DISPATCH(res->device().getType(), res, q, scale, x, weight, epsilon);
}
void Dsv4AddRMSNormQuant::execute(Tensor res, Tensor q, Tensor scale, const Tensor &x, const Tensor &weight, float epsilon) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Dsv4AddRMSNormQuant, res, q, scale, x, weight, epsilon);
}
void dsv4_add_rmsnorm_quant_(Tensor res, Tensor q, Tensor scale, const Tensor &x, const Tensor &weight, float epsilon) { Dsv4AddRMSNormQuant::execute(res, q, scale, x, weight, epsilon); }
} // namespace infinicore::op
