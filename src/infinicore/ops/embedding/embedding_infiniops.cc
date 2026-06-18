#include "infinicore/ops/embedding.hpp"

#ifdef ENABLE_INFINIOPS_API
#include "../infiniops_impl.hpp"

#include "base/embedding.h"

namespace infinicore::op::embedding_impl::infiniops {
namespace {

using TensorMeta = ::infinicore::op::infiniops::TensorMeta;

struct PlannedMeta {
    TensorMeta out, input, weight;
    graph::GraphTensor out_tensor, input_tensor, weight_tensor;
};

} // namespace

void *plan(Tensor out, const Tensor &input, const Tensor &weight) {
    INFINICORE_ASSERT(out->device().getType() == Device::Type::NVIDIA);
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, input, weight);

    return new PlannedMeta{
        TensorMeta(out),
        TensorMeta(input),
        TensorMeta(weight),
        graph::GraphTensor(out),
        graph::GraphTensor(input),
        graph::GraphTensor(weight)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    infini::ops::Handle handle;
    handle.set_stream(context::getStream());
    infini::ops::Config config;

    infini::ops::Embedding::Call(
        handle,
        config,
        planned->input.tensor(planned->input_tensor),
        planned->weight.tensor(planned->weight_tensor),
        planned->out.tensor(planned->out_tensor));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

static bool registered = []() {
    Embedding::plan_dispatcher().registerDevice(Device::Type::NVIDIA, &plan);
    Embedding::run_dispatcher().registerDevice(Device::Type::NVIDIA, &run);
    Embedding::cleanup_dispatcher().registerDevice(Device::Type::NVIDIA, &cleanup);
    return true;
}();

} // namespace infinicore::op::embedding_impl::infiniops
#endif
