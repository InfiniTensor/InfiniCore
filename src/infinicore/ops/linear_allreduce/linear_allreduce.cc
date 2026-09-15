#include "infinicore/ops/linear_allreduce.hpp"
#include "infinicore/device.hpp"
#include "infinicore/ops/distributed/allreduce.hpp"
#include "infinicore/ops/linear.hpp"

#if defined(ENABLE_ASCEND_API)
namespace infinicore::op::linear_allreduce_impl::ascend {
void linear_allreduce_impl(
    Tensor output, Tensor input, Tensor weight,
    std::optional<Tensor> bias, infinicclComm_t communicator);
} // namespace infinicore::op::linear_allreduce_impl::ascend
#endif

namespace infinicore::op {

Tensor linear_allreduce(
    Tensor input, Tensor weight, std::optional<Tensor> bias,
    infinicclReduceOp_t op, infinicclComm_t communicator) {
#if defined(ENABLE_ASCEND_API)
    if (input->device().getType() == Device::Type::ASCEND) {
        Size ndim = input->ndim();
        Size out_features = weight->shape()[0];
        auto out_shape = input->shape();
        out_shape[ndim - 1] = out_features;
        auto out = Tensor::empty(out_shape, input->dtype(), input->device());
        linear_allreduce_impl::ascend::linear_allreduce_impl(
            out, input, weight, bias, communicator);
        return out;
    }
#endif
    auto output = linear(input, weight, bias);
    return distributed::allreduce(output, op, communicator);
}

void linear_allreduce_(
    Tensor output, Tensor input, Tensor weight,
    std::optional<Tensor> bias, infinicclReduceOp_t op,
    infinicclComm_t communicator) {
#if defined(ENABLE_ASCEND_API)
    if (input->device().getType() == Device::Type::ASCEND) {
        linear_allreduce_impl::ascend::linear_allreduce_impl(
            output, input, weight, bias, communicator);
        return;
    }
#endif
    linear_(output, input, weight, bias);
    distributed::allreduce_(output, output, op, communicator);
}

} // namespace infinicore::op
