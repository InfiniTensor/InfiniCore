#include "infinicore/ops/sigmoid.hpp"

#ifdef ENABLE_INFINIOPS_API
#include "../infiniops_impl.hpp"

#include "base/sigmoid_infinilm.h"

namespace infinicore::op::sigmoid_impl::infiniops {
namespace {

using TensorMeta = ::infinicore::op::infiniops::TensorMeta;

struct PlannedMeta {
    TensorMeta output, input;
    graph::GraphTensor output_tensor, input_tensor;
};

} // namespace

void *plan(Tensor output, const Tensor &input) {
    INFINICORE_ASSERT(output->device().getType() == Device::Type::NVIDIA);
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input);

    return new PlannedMeta{
        TensorMeta(output),
        TensorMeta(input),
        graph::GraphTensor(output),
        graph::GraphTensor(input)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    infini::ops::Handle handle;
    handle.set_stream(context::getStream());
    infini::ops::Config config;

    infini::ops::SigmoidInfinilm::Call(
        handle,
        config,
        planned->input.tensor(planned->input_tensor),
        planned->output.tensor(planned->output_tensor));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

static bool registered = []() {
    Sigmoid::plan_dispatcher().registerDevice(Device::Type::NVIDIA, &plan);
    Sigmoid::run_dispatcher().registerDevice(Device::Type::NVIDIA, &run);
    Sigmoid::cleanup_dispatcher().registerDevice(Device::Type::NVIDIA, &cleanup);
    return true;
}();

} // namespace infinicore::op::sigmoid_impl::infiniops
#endif
