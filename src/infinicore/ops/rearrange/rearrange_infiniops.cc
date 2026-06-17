#include "infinicore/ops/rearrange.hpp"

#ifdef ENABLE_INFINIOPS_API
#include "../infiniops_impl.hpp"

#include "base/rearrange_infinilm.h"

namespace infinicore::op::rearrange_impl::infiniops {
namespace {
using TensorMeta = ::infinicore::op::infiniops::TensorMeta;
struct PlannedMeta {
    TensorMeta y, x;
    graph::GraphTensor y_tensor, x_tensor;
};
} // namespace

void *plan(Tensor y, const Tensor &x) {
    INFINICORE_ASSERT(y->device().getType() == Device::Type::NVIDIA);
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(y, x);
    return new PlannedMeta{TensorMeta(y), TensorMeta(x), graph::GraphTensor(y), graph::GraphTensor(x)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    infini::ops::Handle handle;
    handle.set_stream(context::getStream());
    infini::ops::Config config;
    infini::ops::generated_dispatch::CallRearrangeInfinilm(
        handle, config, planned->x.tensor(planned->x_tensor), planned->y.tensor(planned->y_tensor));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

// ReArrange is used by Tensor::copy_from during test result conversion. Keep
// the InfiniCore/InfiniOp implementation active until this adapter gets the same
// CUDA shim treatment as Gemm.
} // namespace infinicore::op::rearrange_impl::infiniops
#endif
