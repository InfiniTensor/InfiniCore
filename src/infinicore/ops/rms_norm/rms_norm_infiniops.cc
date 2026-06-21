#include "infinicore/ops/rms_norm.hpp"

#ifdef ENABLE_INFINIOPS_API
#include "../infiniops_impl.hpp"

#include "base/rms_norm.h"

namespace infinicore::op::rms_norm_impl::infiniops {
namespace {

using TensorMeta = ::infinicore::op::infiniops::TensorMeta;

struct PlannedMeta {
    TensorMeta y, x, weight;
    graph::GraphTensor y_tensor, x_tensor, weight_tensor;
    float epsilon;
};

} // namespace

void *plan(Tensor y, const Tensor &x, const Tensor &weight, float epsilon) {
    INFINICORE_ASSERT(y->device().getType() == Device::Type::NVIDIA);
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(y, x, weight);

    return new PlannedMeta{
        TensorMeta(y),
        TensorMeta(x),
        TensorMeta(weight),
        graph::GraphTensor(y),
        graph::GraphTensor(x),
        graph::GraphTensor(weight),
        epsilon};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    infini::ops::Handle handle;
    handle.set_stream(context::getStream());
    infini::ops::Config config;

    infini::ops::RmsNorm::Call(
        handle,
        config,
        planned->x.tensor(planned->x_tensor),
        planned->weight.tensor(planned->weight_tensor),
        planned->epsilon,
        planned->y.tensor(planned->y_tensor));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

static bool registered = []() {
    RMSNorm::plan_dispatcher().registerDevice(Device::Type::NVIDIA, &plan);
    RMSNorm::run_dispatcher().registerDevice(Device::Type::NVIDIA, &run);
    RMSNorm::cleanup_dispatcher().registerDevice(Device::Type::NVIDIA, &cleanup);
    return true;
}();

} // namespace infinicore::op::rms_norm_impl::infiniops
#endif
